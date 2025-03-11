"""
Created on Tue Dec 5 20:30:27 2023
@author: Shulei Ji
"""

import torch
from model.PPO_Chord import PPO_Chord
from model.PG_Chord import PG_Chord
from model.DQN_Chord import DQN_Chord
from model.BLSTM_Chord_MH import LSTM_Chord
import pickle
import muspy
import os
import random
from argparse import ArgumentParser
from data_process.midi2representation import midi2event
from evaluate_utils import melody2event, chord_revise, chord_transformation, revise_bar, \
    merge_chord, compute_corpus_level_copy, getTrainChordandDur, hist_sim, duration2type
from chord_metrics import compute_metrics
from torch.nn import LSTM

import torch.nn.functional as F



def batch_data_win(datas, condition_window, seq_len):
    '''
    Prepare one batch data (batch size = 1)
    '''
    one_batch = {}
    one_batch['condition'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['note_t'] = {'pitches': [], 'durations': [], 'positions': []}
    one_batch['chords'] = []
    chord_0 = [0] * 20
    chord_0[0] = 1
    chord_tmp = [chord_0]
    one_batch['chords'].append(chord_tmp)
    for t in range(seq_len):
        if t - condition_window / 2 >= 0 and t + condition_window / 2 - 1 < seq_len:
            window_start = int(t - condition_window / 2)
            window_end = int(t + condition_window / 2)
        elif t - condition_window / 2 < 0:
            window_start = 0
            window_end = int(condition_window)
        else:
            window_start = int(seq_len - condition_window)
            window_end = int(seq_len)
        pitch = []
        duration = []
        position = []
        pitch_tt = []
        duration_tt = []
        position_tt = []
        pitch_temp = datas['pitchs'][window_start:window_end]
        for i in range(len(pitch_temp)):
            if pitch_temp[i] != 0:
                pitch_temp[i] -= 47
        pitch.append(pitch_temp)
        d_temp = []
        for hhh in datas['durations'][window_start:window_end]:
            d_temp.append(duration2type(hhh))
        duration.append(d_temp)
        position.append(datas['bars'][window_start:window_end])
        pitch_t = [0] * 49
        duration_t = [0] * 12
        position_t = [0] * 72
        if datas['pitchs'][t] == 0:
            pitch_t[0] = 1
        else:
            pitch_t[datas['pitchs'][t] - 47] = 1
        dur = duration2type(datas['durations'][t])
        duration_t[dur] = 1
        position_t[datas['bars'][t]] = 1
        pitch_tt.append(pitch_t)
        duration_tt.append(duration_t)
        position_tt.append(position_t)
        one_batch['condition']['pitches'].append(pitch)
        one_batch['condition']['durations'].append(duration)
        one_batch['condition']['positions'].append(position)
        one_batch['note_t']['pitches'].append(pitch_tt)
        one_batch['note_t']['durations'].append(duration_tt)
        one_batch['note_t']['positions'].append(position_tt)
    return one_batch


def generate(src, dis, temperature=1.0):
    """
    Generate chord for a given melody (one sample) with temperature-based sampling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition_window = 8
    input_size = 64
    hidden_size = 512

    models = {"PG": PG_Chord,
              "DQN": DQN_Chord,
              "PPO": PPO_Chord,
              "BLSTM": LSTM_Chord}
    model_class = models[args.model]
    model = model_class(condition_window, input_size, hidden_size).to(device)
    
    if args.load_model:
        model_dir = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}"
        load_model_path = os.path.join(model_dir, args.load_model)
    else:
        raise ValueError("Model path must be provided.")
    
    checkpoint = torch.load(load_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    event = melody2event(src)
    seq_len = len(event["bars"])
    one_batch = batch_data_win(event, condition_window, seq_len)
    chord_t_1 = torch.Tensor(one_batch['chords'][0]).to(device)
    hidden = None
    chord_order = []
    
    for i in range(seq_len):
        condition = one_batch['condition']
        note = one_batch['note_t']
        condition_t = {key: torch.LongTensor(condition[key][i]).to(device) for key in condition}
        note_t = {key: torch.Tensor(note[key][i]).to(device) for key in note}
        
        condition_tt = torch.cat([condition_t['pitches'], condition_t['durations'], condition_t['positions']], dim=1).to(device)
        note_tt = torch.cat([note_t['pitches'], note_t['durations'], note_t['positions']], dim=1).to(device)
        state = torch.cat([condition_tt, note_tt, chord_t_1], dim=-1).to(device)
        states = state.view(1, 1, -1)
        
        if args.model in ["PG", "PPO"]:
            output_1, output_2, output_4, output_13, hidden = model(states, hidden)
        elif args.model == "DQN":
            output_1, output_2, output_4, output_13, hidden = model.act(states, hidden)
        elif args.model == "BLSTM":
            output_1, output_2, output_4, output_13, hidden = model(states, hidden)
        
        # Sample chord presence with temperature
        probs_1 = torch.sigmoid(output_1 / temperature)
        topi_1 = torch.bernoulli(probs_1).to(device)
                # Sample chord parameters using softmax + temperature
        output_2 = output_2.squeeze(0)  # Remove batch dimension if necessary
        output_4 = output_4.squeeze(0)
        output_13 = output_13.squeeze(0)

        # Sample chord parameters using softmax + temperature
        topi_2 = torch.multinomial(F.softmax(output_2 / temperature, dim=-1), 1).unsqueeze(0)
        topi_4 = torch.multinomial(F.softmax(output_4 / temperature, dim=-1), 1).unsqueeze(0)
        topi_13 = torch.multinomial(F.softmax(output_13 / temperature, dim=-1), 5).unsqueeze(0)
        
        ctmp = torch.cat((topi_1, topi_2, topi_4, topi_13), dim=-1)[0][0].int().cpu().numpy().tolist()
        
        if ctmp[0] == 1:
            chord_order.append([0])
            chord_0 = [0] * 20
            chord_0[0] = 1
            chord_t_1 = torch.Tensor([chord_0]).to(device)
        else:
            chord_t_1 = torch.zeros(1, 20).to(device)
            chord_t_1[0][1 + ctmp[1]] = 1
            chord_t_1[0][3 + ctmp[2]] = 1
            chord_order_new = chord_transformation(ctmp[1:-1])
            chord_order_new = chord_revise(chord_order_new)
            chord_order.append(chord_order_new)
            for p in chord_order_new[1:]:
                chord_t_1[0][7 + p] = 1
    
    music = muspy.read_midi(src, 'pretty_midi')
    times = music.time_signatures
    chord_order, event["pitchs"], event["bars"], event["durations"] = revise_bar(
        chord_order, event["pitchs"], event["bars"], event["durations"], times
    )
    new_chords, new_durations = merge_chord(chord_order, event["bars"], event["durations"])

    

    # Combine generated chords with the melody into a new MIDI file if requested
    if args.generate_midi:
        notesss = []
        start_time = 0
        for i in range(len(new_chords)):
            chord_t = new_chords[i]
            if len(chord_t) != 1:
                chords = []
                t = 2 if chord_t[0] == 0 else 3
                offset = 0
                for j in range(1, len(chord_t)):
                    if j > 1 and chord_t[j] < chord_t[j - 1]:
                        offset += 1
                    pitch = 12 + (t + offset) * 12 + chord_t[j]
                    notee = muspy.Note(
                        time=start_time, pitch=pitch,
                        duration=new_durations[i],
                        velocity=music.tracks[0].notes[0].velocity
                    )
                    chords.append(notee)
                notesss.extend(chords)
            start_time += new_durations[i]
        
        chord_track = muspy.Track(program=0, is_drum=False, name='', notes=notesss)
        music.tracks.insert(-1, chord_track)
        print(dis)
        print(music)
        muspy.write_midi(dis, music)




def generate_compute_metrics(generate_path):
    '''
    Compute metrics and generate chords for given melodies in a directory.
    '''
    CHS_ALL = []
    CTD_aver = 0
    CTnCTR_aver = 0
    PCS_aver = 0
    MCTD_aver = 0
    CNR_aver = 0
    cnt_64 = 0
    cnt = 0
    midi_files = os.listdir(args.test_data_path)
    random.seed(13)
    random.shuffle(midi_files)
    while cnt < len(midi_files) and cnt_64 < int(args.test_num):
        src_path = os.path.join(args.test_data_path, midi_files[cnt])
        music_length = len(midi2event(src_path)['pitchs'])
        # Only calculate metrics for music with a length larger than 14
        if music_length >= 14:
            dis_path = os.path.join(generate_path, midi_files[cnt])
            CHS, CTD, CTnCTR, PCS, MCTD, CNR = generate(src_path, dis_path)
            print(CHS)
            CHS_ALL.append(CHS)
            CTD_aver += CTD
            CTnCTR_aver += CTnCTR
            PCS_aver += PCS
            MCTD_aver += MCTD
            CNR_aver += CNR
            cnt_64 += 1
        cnt += 1
    CTD_aver /= int(args.test_num)
    CTnCTR_aver /= int(args.test_num)
    PCS_aver /= int(args.test_num)
    MCTD_aver /= int(args.test_num)
    CNR_aver /= int(args.test_num)
    return CHS_ALL, CTD_aver, CTnCTR_aver, PCS_aver, MCTD_aver, CNR_aver


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate RL-Chord')
    # Existing arguments
    parser.add_argument("--dataset", type=str, default='NMD', help="NMD or Wiki")
    parser.add_argument("--seq_len", type=str, default='64', help="64 or 128")
    parser.add_argument("--model", type=str, default='PG', help="PG or DQN or PPO or CF")
    parser.add_argument("--repre", type=str, default='MH', help="MH or GT")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--generate_midi", type=bool, default=1)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--generate_path", type=str, default=None)
    parser.add_argument("--hist_path", type=str, default=None, help="Save histogram for the CHS metric")
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1)

    # New arguments for single inference
    parser.add_argument("--input_path", type=str, default=None, help="Path to input midi file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output midi file")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If input_path and output_path are provided, run single inference and exit.
    if args.input_path is not None and args.output_path is not None:
        generate(args.input_path, args.output_path,args.temperature)
        exit(0)

    # Otherwise, load the model for batch evaluation.
    condition_window = 16 # originally 8
    input_size = 64
    hidden_size = 512
    models = {"PG": PG_Chord,
              "DQN": DQN_Chord,
              "PPO": PPO_Chord,
              "BLSTM": LSTM_Chord}

    model = models[args.model](condition_window, input_size, hidden_size).to(device)
    model_dir = f"./saved_models/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}"
    load_model_path = os.path.join(model_dir, args.load_model) if args.load_model is not None \
                     else None
    checkpoint_dict = torch.load(load_model_path, map_location=device) if load_model_path is not None \
                      else None
    if checkpoint_dict is not None:
        model.load_state_dict(checkpoint_dict['model'])
    model.eval()

    # Load ground truth chord and duration sequences.
    gt_chord_data = f"data/{args.dataset}_GT_chord_order_list.data"
    gt_dur_data = f"data/{args.dataset}_GT_duration_list.data"
    if not os.path.exists(gt_chord_data):
        chord_order_GT_list, duration_GT_list = getTrainChordandDur(args.test_data_path)
        with open(gt_chord_data, 'wb') as file:
            pickle._dump(chord_order_GT_list, file)
        with open(gt_dur_data, 'wb') as file:
            pickle._dump(duration_GT_list, file)
    else:
        with open(gt_chord_data, 'rb') as file:
            GT_chord_list = pickle.load(file)
        with open(gt_dur_data, 'rb') as file:
            GT_dur_list = pickle.load(file)

    # Compute metrics and generate music for batch evaluation.
    corpus_level_copy_num_list = []
    generate_path = f"{args.generate_path}/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}"
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)
    
    CHS_ALL, CTD_aver, CTnCTR_aver, PCS_aver, MCTD_aver, CNR_aver = generate_compute_metrics(generate_path)
    hist_path = f"{args.hist_path}/{args.dataset}-{args.model}-{args.repre}-{args.seq_len}.data"
    with open(hist_path, 'wb') as file:
        pickle._dump(CHS_ALL, file)
    
    print("CNR_aver: ", CNR_aver)
    print("CTD_aver: ", CTD_aver)
    print("DC_aver: ", CTnCTR_aver)
    print("PCS_aver: ", PCS_aver)
    print("MCTD_aver: ", MCTD_aver)
    print("corpus_level_copy_num_list: ", corpus_level_copy_num_list)
