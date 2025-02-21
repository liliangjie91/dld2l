#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os,ffmpeg,configparser,logging
from openai import OpenAI
import logging
from py_logging_config import setup_logging
from tqdm import tqdm


def get_logger_my(logger_name):
    return logging.getLogger(logger_name)

def openai_speech2text_my(audio_file_paths,config_file='config.ini',api_prefix='sk'):
    logger = get_logger_my('my_logger')
    # 读取配置文件
    config = configparser.ConfigParser()
    config.read(config_file)
    api_prefix=api_prefix # 'sk'
    client = OpenAI(
        api_key=config['openai']['api_key_' + api_prefix],
        base_url=config['openai']['api_base_' + api_prefix],
    )
    # 定义每个片段的大小（以字节为单位）
    chunk_size = 1024 * 1024  # 1MB
    for audio_file_path in audio_file_paths:
        logger.info('processing {}'.format(audio_file_path))
        # 读取音频文件并切分
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()
        chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

        # 存储所有片段的转录结果
        transcriptions = []
        # 依次发送每个片段的请求
        for chunk in tqdm(chunks, desc=f"Processing {os.path.basename(audio_file_path)}"):  # 使用 tqdm 包装 chunks
            with open('./tmp/temp_chunk.mp3', 'wb') as temp_file:
                temp_file.write(chunk)
            
            with open('./tmp/temp_chunk.mp3', "rb") as temp_audio_file:
                response = client.audio.transcriptions.create(
                    model="FunAudioLLM/SenseVoiceSmall",
                    file=temp_audio_file,
                    response_format="json"
                )
            
            transcriptions.append(response.text)
        file_name, _ = os.path.splitext(audio_file_path)
        out_text_file_name = file_name + '.txt'
        with open(out_text_file_name, 'w', encoding='utf-8') as temp_file:
            for line in transcriptions:
                temp_file.write(line)
        # logger.info('translated audio file to {}'.format(out_text_file_name))

def vedio2mp3_my(folder_path):
    # 获取文件夹中的所有文件
    logger = get_logger_my('my_logger')
    all_files = os.listdir(folder_path)
    # 过滤出 MP4 文件
    mp4_files = [file for file in all_files if file.lower().endswith('.mp4')]
    mp3_files = []
    for mp4_file in mp4_files:
        file_name, _ = os.path.splitext(mp4_file)
        mp3_file_name = folder_path + '/audio_' + file_name + '.mp3'
        logger.info('transleting {}'.format(file_name))
        ffmpeg.input(folder_path+'/'+mp4_file).output(mp3_file_name, q=0, map='a').run(overwrite_output=True,quiet=True)
        logger.info('saved to {}'.format(mp3_file_name))
        mp3_files.append(mp3_file_name)
    return mp3_files

def get_files(folder_path, file_type='mp4', max_files=0):
    all_files = os.listdir(folder_path)
    # 过滤文件
    if file_type:
        all_files = [file for file in all_files if file.lower().endswith(file_type)]
    if max_files:
        all_files = all_files[:max_files]
    res = [folder_path + '/' + file_name for file_name in all_files]
    print('get {} files'.format(len(res)))
    return res

if __name__ == '__main__':
    # 配置日志记录器
    setup_logging()
    folder_path = "E:/告别原生家庭直通"
    # mp3_files = vedio2mp3_my(folder_path)
    mp3_files = get_files(folder_path, file_type='mp3')
    openai_speech2text_my(mp3_files)
    
    