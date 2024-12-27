import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generation of random quadratic curve data
def generate_curve_data(quadrant):
    np.random.seed(42)  # 再現性のためのseed設定
    
    # ランダムな二次関数の係数 a を生成
    if quadrant in [0, 1, 2]:  # 第1象限と第2象限: 上に開く
        a = np.random.uniform(0.01, 0.1)
    elif quadrant in [3, 4]:  # 第3象限と第4象限: 下に開く
        a = np.random.uniform(-0.01, -0.1)
    
    # ランダムな x 値の生成
    if quadrant in [1, 4]:  # 第1象限と第4象限: x > 0
        x = np.random.uniform(-30, 150, 100)   
    elif quadrant in [2, 3]:  # 第2象限と第3象限: x < 0
        x = np.random.uniform(-150, 10, 100)
    elif quadrant == 0:
        x = np.random.uniform(-50, 50, 100)
    
    # x = np.random.uniform(-20, 150, 100)
    # y 値の計算 (象限ごとに異なる平行移動を適用)
    noise = np.random.normal(-10, 10, size=x.shape)  # ノイズ
    
    theta_rotation = 0
    
    if quadrant == 1:
        vertex = (70, 40)  # 第1象限: y > 0
        theta_rotation = -np.pi / 6
    elif quadrant == 2:
        vertex = (-85, 55)  # 第2象限: y > 0
        theta_rotation = np.pi / 12
    elif quadrant == 3:
        vertex = (-95, -72)  # 第3象限: y < 0
        theta_rotation = -np.pi / 8
    elif quadrant == 4:
        vertex = (70, -100)  # 第4象限: y < 0
        theta_rotation = np.pi / 8
    elif quadrant == 0:
        vertex = (0, 10)
        
    y = a * (x - vertex[0])**2 + vertex[1] + noise
    
    # rotation
    # x, y = rotate_about_vertex(x, y, theta_rotation, vertex)
    
    # # 双曲線
    # if quadrant in [1, 3]:
    #     # x = np.random.uniform(0, 50, 100)
    #     noise = np.random.normal(-2, 2, size=x.shape)  # ノイズ
    #     y = 1 / x + noise
    # elif quadrant in [2, 4]:
    #     # x = np.random.uniform(-50, 0, 100)
    #     noise = np.random.normal(-2, 2, size=x.shape)  # ノイズ
    #     y = -1 / x + noise
        
    # if quadrant == 1:
    #     x = np.random.uniform(10.1, 40, 100)
    #     noise = np.random.normal(-0.5, 0.5, size=x.shape)  # ノイズ
    #     y = 1 / (x - 10) + 15 + noise
    # elif quadrant == 2:
    #     x = np.random.uniform(-50, -20, 100)
    #     noise = np.random.normal(-2, 2, size=x.shape)  # ノイズ
    #     y = -1 / (x + 20) + 10 + noise
    # elif quadrant == 3:
    #     x = np.random.uniform(-45, -15, 100)
    #     noise = np.random.normal(-2, 2, size=x.shape)  # ノイズ
    #     y = 1 / (x + 15) - 20 + noise
    # elif quadrant == 4:
    #     x = np.random.uniform(25, 55, 100)
    #     noise = np.random.normal(-2, 2, size=x.shape)  # ノイズ
    #     y = -1 / (x - 25) - 25 + noise
    
    # log
    # if quadrant == 1:
    #     vertex = (3, 47)  # 第1象限: y > 0
    #     a = -10
    #     b = 1
    #     x = np.random.uniform(3, 200, 100)
    # elif quadrant == 2:
    #     vertex = (10, 55)  # 第2象限: y > 0
    #     a = -10
    #     b = -1
    #     x = np.random.uniform(-200, -10, 100)
    # elif quadrant == 3:
    #     vertex = (-3, -50)  # 第3象限: y < 0
    #     a = 10
    #     b = -1
    #     x = np.random.uniform(-200, -3, 100)
    # elif quadrant == 4:
    #     vertex = (2, -50)  # 第4象限: y < 0
    #     a = 10
    #     b = 1
    #     x = np.random.uniform(2, 200, 100)
    # elif quadrant == 0:
    #     vertex = (0, 10)
        
    # y = a * np.log(b * x - vertex[0]) + vertex[1] + noise
    
    return x, y

def fit_quadratic_curve(x_data, y_data):
    # 制約条件
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = y_data[np.argmin(x_data)], y_data[np.argmax(x_data)]  # x_min, x_max に対応する y

    # 残りのデータ点（最小値と最大値を除いたデータ）
    x_rest = x_data[(x_data > x_min) & (x_data < x_max)]
    y_rest = y_data[(x_data > x_min) & (x_data < x_max)]

    # c を a, b に置き換える
    def calc_c(a, b):
        return y_min - a * x_min**2 - b * x_min

    # b を a に置き換える
    def calc_b(a):
        return (y_max - y_min - a * (x_max**2 - x_min**2)) / (x_max - x_min)

    # 最小二乗法で a を最適化
    def loss(a):
        b = calc_b(a)
        c = calc_c(a, b)
        residuals = y_rest - (a * x_rest**2 + b * x_rest + c)  # 残差
        return np.sum(residuals**2)

    # 最適な a を探す
    result = minimize(loss, x0=0)  # 初期値を a=0 とする
    a_opt = result.x[0]

    # 最適な b, c を計算
    b_opt = calc_b(a_opt)
    c_opt = calc_c(a_opt, b_opt)

    # 結果の二次関数
    return a_opt, b_opt, c_opt


def convert_function(x, y, theta, dx, dy):
    # 回転行列
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # 座標を回転
    coords = np.vstack((x, y))
    rotated_coords = rotation_matrix @ coords
    
    # 平行移動
    x_rotated_translated = rotated_coords[0, :] + dx
    y_rotated_translated = rotated_coords[1, :] + dy
    
    return x_rotated_translated, y_rotated_translated


def calculate_vertex(a, b, c):
    if a == 0:
        raise ValueError("This is a straight line, not a quadratic function.")
    
    # x-coordinate of the vertex
    x_vertex = -b / (2 * a)
    # y-coordinate of vertex
    y_vertex = a * x_vertex**2 + b * x_vertex + c
    
    return x_vertex, y_vertex


def rotate_about_vertex(x, y, theta, vertex):
    # Translate to make the vertex the origin
    x_translated = x - vertex[0]
    y_translated = y - vertex[1]
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Rotate the translated points
    coords = np.vstack((x_translated, y_translated))
    rotated_coords = rotation_matrix @ coords
    
    # Translate back to the original position
    x_rotated = rotated_coords[0, :] + vertex[0]
    y_rotated = rotated_coords[1, :] + vertex[1]
    
    return x_rotated, y_rotated


def draw_line_through_points(p1, p2):
    """
    2点を通る直線を描画するための座標を計算
    p1, p2: (x, y) 座標のタプル
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # 十分に長い線分を描画するため、線を延長
    if abs(x2 - x1) < 1e-10:  # ほぼ垂直な線の場合
        x = np.array([x1, x1])
        y = np.array([y1 - 100, y1 + 100])  # 適当な長さで延長
    else:
        # 直線の方程式: y = mx + b
        m = (y2 - y1) / (x2 - x1)  # 傾き
        b = y1 - m * x1  # y切片
        
        # x座標の範囲を広げて線を延長
        x_min = min(x1, x2) - 150
        x_max = max(x1, x2) + 150
        x = np.array([x_min, x_max])
        y = m * x + b
    
    return x, y


def main():
    np.random.seed(42)  # 再現性のためのseed設定
    # 全象限データを生成してプロット
    plt.figure(figsize=(10, 10))
    
    # debug draw circle
    r_c = 30
    theta_ = np.linspace(0, 2 * np.pi, 1000)
    x = r_c * np.cos(theta_)
    y = r_c * np.sin(theta_)

    plt.plot(x, y, label='Unit Circle', color="green", linewidth=2)
    
    vertexs = []
    
    for quadrant in range(1, 5):
        # Generation of random quadratic curve data
        x_data, y_data = generate_curve_data(quadrant=quadrant)
        
        # plot
        plt.scatter(x_data, y_data, label=f"Noisy Data_{quadrant}", color="blue", alpha=0.6)
        
        "Fit quadratic curve"
        # fitting
        a_opt, b_opt, c_opt = fit_quadratic_curve(x_data, y_data)
        x_vertex, y_vertex = calculate_vertex(a_opt, b_opt, c_opt)
        
        print(f"Constrained least-squares solution: y = {a_opt:.4f}x^2 + {b_opt:.4f}x + {c_opt:.4f}")
        
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = a_opt * x_fit**2 + b_opt * x_fit + c_opt

        plt.plot(x_fit, y_fit, color="red", label="Constrained Quadratic Fit")
        # plt.scatter(x_vertex, y_vertex, label="vertex", color="black", alpha=0.6)
        
        
        # 対称軸の長さを十分に取る
        line_length = 100
        # 頂点から上下に伸びる垂直な線を生成
        x_sym = np.array([x_vertex, x_vertex])
        y_sym = np.array([y_vertex - line_length, y_vertex + line_length])
        # plt.plot(x_sym, y_sym, '--', color='green', alpha=0.7, label='Symmetry Line')
        
        "Rotation and translation"
        # rotation
        # theta_rotation = np.pi / 6  # 30°
        theta_rotation = np.random.uniform(-np.pi / 4, np.pi / 6)
        if quadrant == 4:
            theta_rotation = np.random.uniform(0, np.pi / 3)
        x_rotated, y_rotated = convert_function(x_fit, y_fit, theta_rotation, 0, 0)
        x_vertex_rotated, y_vertex_rotated = convert_function(x_vertex, y_vertex, theta_rotation, 0, 0)
        
        x_sym_rotated, y_sym_rotated = convert_function(x_sym, y_sym, theta_rotation, 0, 0)
        # plt.plot(x_sym_rotated, y_sym_rotated, '--', color='green', alpha=0.7, label='rotated Symmetry Line')
        
        # plt.plot(x_rotated, y_rotated, color="green", label="rotated")
        # plt.scatter(x_vertex_rotated, y_vertex_rotated, label="vertex_rotated", color="black", alpha=0.6)
        
        # translation
        # theta_c = np.pi * 2 / 3
        theta_c = np.pi / 4 + (np.pi / 2) * (quadrant + 3) * np.random.uniform(0.95, 1.05) + np.random.uniform(-np.pi / 12, np.pi / 12)
        dx_c = r_c * np.cos(theta_c) - x_vertex_rotated
        dy_c = r_c * np.sin(theta_c) - y_vertex_rotated
        
        x_on_circle, y_on_circle = convert_function(x_rotated, y_rotated, 0, dx_c, dy_c)
        x_vertex_on_circle, y_vertex_on_circle = convert_function(x_vertex_rotated, y_vertex_rotated, 0, dx_c, dy_c)
        
        x_sym_on_circle, y_sym_on_circle = convert_function(x_sym_rotated, y_sym_rotated, 0, dx_c, dy_c)
        # plt.plot(x_sym_on_circle, y_sym_on_circle, '--', color='green', alpha=0.7, label='On circle Symmetry Line')
        
        # plt.plot(x_on_circle, y_on_circle, color="green", label="On circle")
        plt.scatter(x_vertex_on_circle, y_vertex_on_circle, label="vertex on circle", color="black", alpha=0.6)

        # Rotation about the vertex
        # theta_v = np.pi / 6
        # np.random.seed()
        if quadrant % 2 == 0:
            theta_v = np.random.uniform(np.pi / 6, np.pi / 4)
        else:
            theta_v = (-1) * np.random.uniform(np.pi / 6, np.pi / 4)
        x_rotated_v, y_rotated_v = rotate_about_vertex(x_on_circle, y_on_circle, theta_v, (x_vertex_on_circle, y_vertex_on_circle))
        
        plt.plot(x_rotated_v, y_rotated_v, color="purple", label="Rotation about the vertex")
        
        x_sym_rotated_v, y_sym_rotated_v = rotate_about_vertex(x_sym_on_circle, y_sym_on_circle, theta_v, (x_vertex_on_circle, y_vertex_on_circle))
        # plt.plot(x_sym_rotated_v, y_sym_rotated_v, '--', color='green', alpha=0.7, label='On circle Symmetry Line')
        
        vertexs.append((x_vertex_on_circle, y_vertex_on_circle))
    
    x_line, y_line = draw_line_through_points(vertexs[0], vertexs[2])
    plt.plot(x_line, y_line, '--', color='darkred', alpha=0.7, label='Line through points')
    x_line, y_line = draw_line_through_points(vertexs[1], vertexs[3])
    plt.plot(x_line, y_line, '--', color='darkred', alpha=0.7, label='Line through points')
    
    
    
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    # plt.xlim(-30,30)
    # plt.ylim(-30,30)
    # plt.xlim(-150,150)
    # plt.ylim(-150,150)
    plt.xlim(-200,200)
    plt.ylim(-200,200)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    main()