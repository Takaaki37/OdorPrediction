# OpenCVを使用して画像を表示し、右クリックした位置の座標を記録する
import cv2

# 右クリックイベントの座標を保存するリスト
right_clicks = list()

# マウスが右クリックされた時に呼び出される関数
def mouse_callback(event, x, y, flags, params):
    # 右クリックイベントの値は2
    if event == cv2.EVENT_RBUTTONDOWN:  # OpenCVのイベントを直接参照
        # 座標をリストに追加
        right_clicks.append([x, y])
        # マウスデータが収集されていることを確認（デバッグ用、最終的には削除推奨）
        print(right_clicks)

# 画像ファイルとパス
imageFile = "image.jpg"
path = "path"

# 画像を読み込み
img = cv2.imread(path + "/" + imageFile)
# 画像のサイズを1000ピクセルに収まるように調整
scale_width = 1000 / img.shape[1]
scale_height = 1000 / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

# ウィンドウの設定（サイズ可変）
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

# マウスコールバック関数の設定
cv2.setMouseCallback('image', mouse_callback)

# 画像の表示
cv2.imshow('image', img)
cv2.waitKey(0)  # 何かキーが押されるまで待機
cv2.destroyAllWindows()  # すべてのウィンドウを閉じる
