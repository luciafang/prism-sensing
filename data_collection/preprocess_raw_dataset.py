"""
Preprocess the raw dataset to be used in the task.

Before:
- datadrive / tasks / {task_name} / dataset / raw

After:
- datadrive / tasks / {task_name} / dataset / original
"""

import argparse
import librosa
import os
import pandas as pd
import soundfile
from scipy.io import wavfile


from prism import config
from prism.har import params

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--pid', type=str, help='Participant ID', required=True)
    parser.add_argument('--no-video-export', action='store_true', help='Do not export video')
    parser.add_argument('--no-sensor-input', action='store_true', help='Do not use sensor input')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_dir = config.datadrive / 'tasks' / args.task / 'dataset'
    raw_dir = dataset_dir / 'raw' / args.pid
    original_dir = dataset_dir / 'original' / args.pid
    original_dir.mkdir(parents=True, exist_ok=True)

    
    if not args.no_sensor_input:
        # audio
        with open(raw_dir / 'clap_time_audio.txt', 'r') as f:
            clap_time_ms = float(f.readline().strip())
        # assert 1 == 2, 'We have to check different audio file format (librosa vs wavfile)'
        audio, _ = librosa.load(raw_dir / 'audio.wav', sr=params.SAMPLE_RATE)
        trimmed_audio = audio[int(clap_time_ms * params.SAMPLE_RATE / 1000):]
        soundfile.write(original_dir / 'audio.wav', trimmed_audio, samplerate=params.SAMPLE_RATE)

        # motion
        sensor_columns = ['unix_time', 'data.userAcceleration.x', 'data.userAcceleration.y', 'data.userAcceleration.z',
                        'data.gravity.x', 'data.gravity.y', 'data.gravity.z',
                        'data.rotationRate.x', 'data.rotationRate.y', 'data.rotationRate.z',
                        'data.magneticField.field.x', 'data.magneticField.field.y', 'data.magneticField.field.z',
                        'data.attitude.roll', 'data.attitude.pitch', 'data.attitude.yaw',
                        'data.attitude.quaternion.x', 'data.attitude.quaternion.y', 'data.attitude.quaternion.z',
                        'data.attitude.quaternion.w', 'data.time']
        motion = pd.read_csv(raw_dir / 'motion.txt', sep='\s', engine='python', index_col=False)
        if motion.shape[1] - len(sensor_columns) == 1:  # this is an old version of the motion data (nan is included)
            motion = motion.dropna(axis=1)
        assert motion.shape[1] == len(sensor_columns)
        motion.columns = sensor_columns
        trimmed_motion = pd.DataFrame()
        trimmed_motion['timestamp'] = motion['data.time']  # use the sensor timestamp (this is in sec)
        trimmed_motion['acc.x'] = - (motion['data.userAcceleration.x'] + motion['data.gravity.x']) * 9.81
        trimmed_motion['acc.y'] = - (motion['data.userAcceleration.y'] + motion['data.gravity.y']) * 9.81
        trimmed_motion['acc.z'] = - (motion['data.userAcceleration.z'] + motion['data.gravity.z']) * 9.81
        ## crop
        trimmed_motion['timestamp'] *= 1000
        trimmed_motion['timestamp'] -= trimmed_motion['timestamp'].to_list()[0]
        trimmed_motion['timestamp'] -= clap_time_ms
        trimmed_motion = trimmed_motion[trimmed_motion['timestamp'] >= 0]
        trimmed_motion.to_csv(original_dir / 'motion.txt', index=False, sep=' ')

    # annotation
    annotation = pd.read_csv(raw_dir / 'via_annotation_video.csv')
    tmp_annotation_dict = {}
    for _, row in annotation.iterrows():
        start_time_sec = float(row['temporal_coordinates'].split(',')[0][1:])
        label = row['metadata'].split('\"')[3]
        tmp_annotation_dict[label] = start_time_sec * 1000
    annotation_dict = {'Timestamp': [], 'Step': []}
    for i, (label, time) in enumerate(sorted(tmp_annotation_dict.items(), key=lambda x: x[1])):
        if i == 0:
            assert label == 'clap'
            annotation_dict['Timestamp'].append(0)
            annotation_dict['Step'].append('BEGIN')
        else:
            annotation_dict['Timestamp'].append(time - tmp_annotation_dict['clap'])
            annotation_dict['Step'].append(label)
    annotation = pd.DataFrame(annotation_dict)
    annotation.to_csv(original_dir / 'annotation.txt', index=False)

    if not args.no_video_export:
        # video
        video_path = raw_dir / 'video.mp4'
        video_output_path = original_dir / 'video.mp4'
        print(tmp_annotation_dict)
        os.system(f'ffmpeg -ss {tmp_annotation_dict["clap"] / 1000} -i {video_path} -to {tmp_annotation_dict["END"] / 1000} -c:v copy -c:a copy {video_output_path}')
    