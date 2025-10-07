import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

# --- イージング関数 ---
def ease_in_out_quad(t):
    """
    イーズイン・イーズアウトの計算式。tは0.0から1.0の進捗度。
    ゆっくり始まって加速し、最後にゆっくり減速するような値を返す。
    """
    if t < 0.5:
        return 2 * t * t
    return (-2 * t * t) + (4 * t) - 1

# --- 関数の定義（load_and_prepare_images, generate_sample_images, get_target_mapping は変更なし） ---
def load_and_prepare_images(path_a, path_b, size=(100, 100)):
    try:
        img_a_gray = Image.open(path_a).resize(size).convert('L')
        img_b_rgb = Image.open(path_b).resize(size).convert('RGB')
    except FileNotFoundError as e:
        print(f"エラー: 画像 '{e.filename}' が見つかりません。")
        print("代わりにサンプル画像 'target.png' と 'input.png' を生成します。")
        generate_sample_images(size)
        img_a_gray = Image.open("target.png").resize(size).convert('L')
        img_b_rgb = Image.open("input.png").resize(size).convert('RGB')
    return np.array(img_a_gray), np.array(img_b_rgb)

def generate_sample_images(size):
    w, h = size
    target_img = Image.new('L', size)
    for y in range(h):
        for x in range(w):
            target_img.putpixel((x, y), int((x / w) * 255))
    target_img.save("target.png")

    input_img = Image.new('RGB', size, color=(255, 200, 200))
    draw = ImageDraw.Draw(input_img)
    draw.ellipse([(w//4, h//4), (w*3//4, h*3//4)], fill=(0, 100, 255))
    input_img.save("input.png")

def get_target_mapping(img_a_gray, img_b_rgb):
    h, w = img_b_rgb.shape[:2]
    img_b_gray = np.dot(img_b_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    b_flat_gray = img_b_gray.flatten()
    a_flat_gray = img_a_gray.flatten()
    b_sorted_indices = np.argsort(b_flat_gray)
    a_sorted_indices = np.argsort(a_flat_gray)
    target_map = np.zeros_like(b_sorted_indices)
    target_map[b_sorted_indices] = a_sorted_indices
    num_pixels = h * w
    initial_indices = np.arange(num_pixels)
    initial_y = initial_indices // w
    initial_x = initial_indices % w
    initial_positions = np.stack((initial_y, initial_x), axis=1)
    target_indices = target_map[initial_indices]
    target_y = target_indices // w
    target_x = target_indices % w
    target_positions = np.stack((target_y, target_x), axis=1)
    pixel_colors = img_b_rgb.reshape(-1, 3)
    return pixel_colors, initial_positions, target_positions

# --- シミュレーション関数 (変更なし) ---
def run_simulation(positions, target_positions, max_frames):
    """ピクセル移動のシミュレーションを行い、全フレームの位置履歴を返す"""
    print("ピクセル移動のシミュレーションを開始します（収束または最大フレームまで）...")
    positions_history = [np.copy(positions)]
    
    with tqdm(total=max_frames) as pbar:
        for frame in range(max_frames):
            if np.all(positions == target_positions):
                print(f"\n収束しました！ ({frame} フレーム)")
                pbar.n = pbar.total
                pbar.refresh()
                break
            
            move_requests = {}
            for i in range(len(positions)):
                pos = positions[i]
                target_pos = target_positions[i]
                if np.array_equal(pos, target_pos):
                    dest_pos = tuple(pos)
                else:
                    dy, dx = target_pos - pos
                    next_pos_candidate = np.copy(pos)
                    if abs(dy) >= abs(dx):
                        next_pos_candidate[0] += np.sign(dy)
                    else:
                        next_pos_candidate[1] += np.sign(dx)
                    dest_pos = tuple(next_pos_candidate)
                
                if dest_pos not in move_requests: move_requests[dest_pos] = []
                move_requests[dest_pos].append(i)

            next_positions = np.copy(positions)
            for dest_pos, candidates in move_requests.items():
                if len(candidates) == 1:
                    next_positions[candidates[0]] = dest_pos
                else:
                    distances = [np.sum(np.abs(positions[j] - target_positions[j])) for j in candidates]
                    winner_idx = candidates[np.argmax(distances)]
                    next_positions[winner_idx] = dest_pos

            np.copyto(positions, next_positions)
            positions_history.append(np.copy(positions))
            pbar.update(1)

    return positions_history

# --- イージング適用関数 (新規) ---
def apply_easing_to_history(history, num_output_frames):
    """シミュレーション履歴にイージングを適用し、新しいフレームリストを生成する"""
    eased_history = []
    original_num_frames = len(history)
    for i in range(num_output_frames):
        # 0.0から1.0の進捗度を計算
        progress = i / (num_output_frames - 1)
        # イージング関数で進捗度を変換
        eased_progress = ease_in_out_quad(progress)
        # 元の履歴のどのフレームに対応するかを計算
        original_frame_index = int(eased_progress * (original_num_frames - 1))
        eased_history.append(history[original_frame_index])
    return eased_history

# --- アニメーション更新関数 (変更なし) ---
def update_animation(frame_num, positions_history, pixel_colors, shape, im, pbar):
    """履歴データに基づいてアニメーションの各フレームを描画する"""
    current_positions = positions_history[frame_num]
    
    new_frame_data = np.zeros(shape, dtype=np.uint8)
    new_frame_data[current_positions[:, 0], current_positions[:, 1]] = pixel_colors
    
    im.set_array(new_frame_data)
    pbar.update(1)
    return im,

# --- メイン処理 (修正) ---
def main():
    # --- パラメータ設定 ---
    TARGET_IMAGE_PATH = 'target.jpg'
    INPUT_IMAGE_PATH = 'input.jpg'
    IMAGE_SIZE = (256, 256)
    OUTPUT_GIF_PATH = 'morphing.gif'
    MAX_FRAMES = 1000 # シミュレーションの最大フレーム数
    
    # ★イージング後のアニメーションの総フレーム数
    ANIMATION_FRAMES = 600
    # ★アニメーションのFPS (フレームレート)
    ANIMATION_FPS = 30

    # 1. 画像の準備
    print("画像を読み込み、前処理を開始します...")
    img_a_gray, img_b_rgb = load_and_prepare_images(TARGET_IMAGE_PATH, INPUT_IMAGE_PATH, IMAGE_SIZE)

    # 2. 移動マッピングの計算
    print("ピクセルの移動先マッピングを計算します...")
    pixel_colors, initial_positions, target_positions = get_target_mapping(img_a_gray, img_b_rgb)
    
    # 3. シミュレーションの実行
    positions_history = run_simulation(np.copy(initial_positions), target_positions, MAX_FRAMES)
    
    # 4. ★シミュレーション履歴にイージングを適用
    print(f"\nシミュレーション結果にイージングを適用し、{ANIMATION_FRAMES}フレームのアニメーションを生成します...")
    eased_positions_history = apply_easing_to_history(positions_history, ANIMATION_FRAMES)

    # 5. アニメーションの準備
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.tight_layout(pad=0)
    im = ax.imshow(img_b_rgb, animated=True)
    
    # 6. アニメーションの生成と保存
    print(f"GIF画像を '{OUTPUT_GIF_PATH}' として保存します...")
    
    with tqdm(total=ANIMATION_FRAMES) as pbar:
        ani = animation.FuncAnimation(
            fig,
            update_animation,
            # ★イージング適用後の履歴を使う
            frames=ANIMATION_FRAMES,
            fargs=(eased_positions_history, pixel_colors, img_b_rgb.shape, im, pbar),
            interval=1000/ANIMATION_FPS, # intervalはミリ秒指定
            blit=True
        )
        ani.save(OUTPUT_GIF_PATH, writer='pillow', fps=ANIMATION_FPS)

    plt.close()
    print(f"処理が完了しました！ '{OUTPUT_GIF_PATH}' を確認してください。")

if __name__ == '__main__':
    main()