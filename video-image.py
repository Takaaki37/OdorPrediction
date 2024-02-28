
# 指定されたビデオファイルからフレームを抽出して保存する

import cv2
import os

def save_frame_range(video_path, start_frame, stop_frame, step_frame, dir_path, basename, ext='jpg'):
    """
    ビデオファイルから指定された範囲のフレームを抽出して保存する関数。
    
    Parameters:
    - video_path: ビデオファイルのパス
    - start_frame: 開始フレーム番号
    - stop_frame: 終了フレーム番号
    - step_frame: フレームを抽出する間隔
    - dir_path: 保存するディレクトリのパス
    - basename: 保存するファイルのベース名
    - ext: 保存するファイルの拡張子（デフォルトは'jpg'）
    """
    cap = cv2.VideoCapture(video_path)  # ビデオを読み込み

    if not cap.isOpened():  # ビデオが開けなかった場合は終了
        return

    os.makedirs(dir_path, exist_ok=True)  # 保存ディレクトリを作成（既に存在する場合は作成しない）
    base_path = os.path.join(dir_path, basename)  # 保存するファイルの基本パス

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))  # フレーム数に合わせたゼロ埋めの桁数を計算

    for n in range(start_frame, stop_frame, step_frame):  # 指定された範囲とステップでループ
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)  # 抽出するフレーム位置を設定
        ret, frame = cap.read()  # フレームを読み込み
        if ret:  # フレームが正常に読み込まれた場合
            # ファイル名にフレーム番号を付けて保存
            cv2.imwrite(f'{base_path}_{str(n).zfill(digit)}.{ext}', frame)
        else:  # フレームの読み込みに失敗した場合は終了
            return

# ビデオファイルからフレームを抽出して保存する関数の呼び出し例
save_frame_range('MOV.mp4', 0, 10000, 1, 'result2', 'frame')


