from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from itertools import combinations, product
G3_states = [
        '00+',  # state1
        '01-',  # state2
        '10-',  # state3
        '11+',  # state4
        '0+0',  # state5
        '0-1',  # state6
        '1-0',  # state7
        '1+1',  # state8
        '+00',  # state9
        '-01',  # state10
        '-10',  # state11
        '+11'   # state12
    ]
psi = (1 / (2 * np.sqrt(2))) * np.array([
        1,  # |000⟩
        1,  # |001⟩
        1,  # |010⟩
        -1, # |011⟩
        1,  # |100⟩
        -1, # |101⟩
        -1, # |110⟩
        -1  # |111⟩
    ], dtype=np.complex128)
def convert_state_to_vector(state_string: str) -> np.ndarray:
    """將字符串形式的量子態轉換為向量"""
    vector_dict = {
        '0': np.array([1, 0], dtype=np.float64),
        '1': np.array([0, 1], dtype=np.float64),
        '+': np.array([1, 1], dtype=np.float64) / np.sqrt(2),
        '-': np.array([1, -1], dtype=np.float64) / np.sqrt(2)
    }
    
    result = vector_dict[state_string[0]]
    for bit in state_string[1:]:
        result = np.kron(result, vector_dict[bit])
    return result

def measurement_operators() -> Tuple[List[np.ndarray], List[str]]:
    """定義所有可能的測量基底 I, X, Y, Z 的組合"""
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # 所有基底的字典
    operators = [I, X, Y, Z]
    operator_labels = ['I', 'X', 'Y', 'Z']
    
    # 生成所有可能的基底組合
    measurements = []
    labels = []
    for op1, op2, op3 in product(operators, repeat=3):
        measurements.append(np.kron(np.kron(op1, op2), op3))
        
        # 使用 np.array_equal 來獲取索引
        label = ''
        for op in [op1, op2, op3]:
            for idx, operator in enumerate(operators):
                if np.array_equal(op, operator):
                    label += operator_labels[idx]
                    break
        labels.append(label)
    
    return measurements, labels

def calculate_expectation_value(density_matrix: np.ndarray, measurement: np.ndarray) -> float:
    """計算期望值"""
    return np.real(np.trace(measurement @ density_matrix))  # 確保返回標量

def calculate_density_matrix_from_state() -> np.ndarray:
    """根據給定的態向量計算密度矩陣"""
    # 計算密度矩陣
    density_matrix = np.outer(psi, psi.conj())
    return density_matrix

def calculate_WG(alpha: float) -> float:
    """計算 <W_G>"""
    measurements, labels = measurement_operators()
    density_matrix = calculate_density_matrix_from_state()  # 使用新的密度矩陣

    # 計算期望值
    expectation_values = [calculate_expectation_value(density_matrix, measurement) for measurement in measurements]
    

    
    EV = 0
    for i, label in zip(expectation_values, labels):
        if not np.isclose(i, 0):  # 檢查期望值是否不等於0
            print(f"不等於0的期望值: {i}, 對應的測量基底: {label}")

        EV += i

    print(f"所有期望值: {expectation_values}")
    
    # 計算 <W_G>
    WG = alpha - (1/4) * EV
    return WG

# 在主函數中調用計算
if __name__ == "__main__":
    alpha_value = 0.0  # 設置 alpha 的值
    WG_value = calculate_WG(alpha_value)
    print(f"<W_G> 的計算結果: {WG_value:.4f}")

density_matrix = calculate_density_matrix_from_state()

# 檢查自伴性
is_hermitian = np.allclose(density_matrix, density_matrix.conj().T)

# 檢查跡
trace = np.trace(density_matrix)

# 檢查特徵值
eigenvalues = np.linalg.eigvals(density_matrix)
print(f"特徵值: {eigenvalues}")

print(f"自伴性: {is_hermitian}")
print(f"跡: {trace}")

norm = np.linalg.norm(psi)
print(f"態向量的範數: {norm}")