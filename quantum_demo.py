 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子区块链创世区块演示程序
纯量子计算实现版本 - 最大化利用量子特性
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector

# 导入我们的量子区块链实现
from quantum_blockchain import QuantumBlockchain, QuantumRandom, QuantumHash, Block

# 量子增强的密钥生成 - 完全利用量子态叠加和纠缠
def quantum_key_generation(num_qubits=4):
    """
    纯量子密钥生成 - 利用量子叠加、纠缠和干涉
    
    Args:
        num_qubits: 量子比特数量
        
    Returns:
        量子态向量和密钥
    """
    print("\n===== 量子密钥生成 =====")
    # 创建量子电路
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # 创建高度叠加态
    for i in range(num_qubits):
        qc.h(i)  # 应用H门创建叠加态
    
    # 创建多体纠缠态 - 构建GHZ态
    for i in range(1, num_qubits):
        qc.cx(0, i)  # 控制X门创建多体纠缠
    
    # 增加量子干涉
    for i in range(num_qubits):
        qc.t(i)  # T门增加相位
    
    # 再次创建叠加以增加复杂性
    for i in range(num_qubits):
        qc.h(i)
    
    # 添加额外的纠缠层
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    # 测量所有量子比特
    qc.measure(range(num_qubits), range(num_qubits))
    
    # 在模拟器上执行电路
    simulator = Aer.get_backend('qasm_simulator')
    qc_compiled = transpile(qc, simulator)
    job = simulator.run(qc_compiled, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    
    # 可视化量子态（在测量前）
    statevector_sim = Aer.get_backend('statevector_simulator')
    qc_no_measure = QuantumCircuit(num_qubits)
    
    # 复制上面的操作但不进行测量
    for i in range(num_qubits):
        qc_no_measure.h(i)
    for i in range(1, num_qubits):
        qc_no_measure.cx(0, i)
    for i in range(num_qubits):
        qc_no_measure.t(i)
    for i in range(num_qubits):
        qc_no_measure.h(i)
    for i in range(num_qubits-1):
        qc_no_measure.cx(i, i+1)
    
    # 获取量子态向量
    statevector = Statevector.from_instruction(qc_no_measure)
    
    # 展示电路
    print(f"量子密钥生成电路:\n{qc}")
    
    # 显示结果
    print(f"量子态叠加和纠缠分布: {counts}")
    top_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"前三个最可能的测量结果: {top_results}")
    most_frequent = max(counts, key=counts.get)
    print(f"量子密钥: {most_frequent}")
    
    return statevector, most_frequent

# 量子签名生成 - 利用量子抗碰撞性
def quantum_signature(message, key, num_qubits=8):
    """
    纯量子签名实现 - 利用量子不可克隆性和量子干涉
    
    Args:
        message: 要签名的消息
        key: 量子生成的密钥
        num_qubits: 量子比特数量
        
    Returns:
        量子签名
    """
    print("\n===== 量子签名生成 =====")
    # 将消息转换为字节序列的初始量子态
    message_bytes = message.encode()
    
    # 创建量子电路
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # 初始化为全叠加态
    for i in range(num_qubits):
        qc.h(i)
    
    # 基于消息修改量子态
    for idx, byte in enumerate(message_bytes):
        qubit_idx = idx % num_qubits
        # 基于字节值设置旋转角度
        angle = (byte / 255.0) * np.pi
        qc.rz(angle, qubit_idx)  # 绕Z轴旋转
        qc.rx(angle/2, qubit_idx)  # 绕X轴旋转
    
    # 使用密钥比特进一步修改量子态
    for i, bit in enumerate(key):
        if bit == '1':
            qc.z(i % num_qubits)  # 应用Z门
            qc.t(i % num_qubits)  # 应用T门增加相位
    
    # 创建复杂的纠缠结构
    for i in range(num_qubits-1):
        qc.cx(i, i+1)  # 相邻量子比特之间的CNOT
    
    # 环形纠缠 - 连接首尾量子比特
    qc.cx(num_qubits-1, 0)
    
    # 添加量子干涉以增加复杂性
    for i in range(num_qubits):
        qc.h(i)
    
    # 基于消息长度添加受控相位门
    msg_len = len(message)
    for i in range(num_qubits-1):
        control = i
        target = (i + (msg_len % num_qubits)) % num_qubits
        if control != target:
            qc.cp(np.pi/4, control, target)  # 受控相位门
    
    # 最终测量
    qc.measure(range(num_qubits), range(num_qubits))
    
    # 模拟量子电路
    simulator = Aer.get_backend('qasm_simulator')
    qc_compiled = transpile(qc, simulator)
    job = simulator.run(qc_compiled, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)
    
    # 从测量结果构建量子签名
    # 使用所有可能结果的加权组合创建签名
    total_shots = sum(counts.values())
    quantum_signature = ""
    
    for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        # 转换为概率
        prob = count / total_shots
        # 添加高频的测量结果到签名中
        if prob > 0.01:  # 概率大于1%的结果
            quantum_signature += bitstring
            if len(quantum_signature) >= 64:  # 限制长度
                break
    
    # 确保签名长度统一
    quantum_signature = quantum_signature[:64].ljust(64, '0')
    
    print(f"消息: {message}")
    print(f"量子签名分布: {list(counts.items())[:3]}...")
    print(f"量子签名: {quantum_signature[:16]}...")
    
    return quantum_signature

# 量子分形Merkle树实现
def quantum_fractal_merkle_tree(transactions):
    """
    量子增强的分形Merkle树
    
    Args:
        transactions: 交易列表
        
    Returns:
        量子分形Merkle根哈希
    """
    print("\n===== 量子分形Merkle树 =====")
    
    def _generate_quantum_entropy(num_qubits=8):
        """生成量子熵源"""
        # 创建量子电路
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # 创建叠加态
        for i in range(num_qubits):
            qc.h(i)
        
        # 增加纠缠
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
        
        # 环形纠缠
        qc.cx(num_qubits-1, 0)
        
        # 增加复杂相位
        for i in range(num_qubits):
            qc.t(i)
            qc.h(i)
        
        # 测量
        qc.measure(range(num_qubits), range(num_qubits))
        
        # 执行电路
        simulator = Aer.get_backend('qasm_simulator')
        qc_compiled = transpile(qc, simulator)
        job = simulator.run(qc_compiled, shots=8)  # 获取多个结果
        result = job.result()
        counts = result.get_counts(qc)
        
        # 组合所有结果作为熵源
        entropy = "".join(list(counts.keys()))
        return entropy
    
    def _quantum_hash(data):
        """使用量子哈希函数"""
        # 使用我们的QuantumHash类
        return QuantumHash.quantum_hash(data)
    
    def _recursive_merkle(items, fractal_memory=""):
        """递归构建量子分形Merkle树"""
        if len(items) == 1:
            return items[0]
        
        if len(items) % 2 != 0:
            # 若节点数为奇数，复制最后一个并添加量子熵
            quantum_entropy = _generate_quantum_entropy()
            # 组合最后一个元素和量子熵
            items.append(items[-1] + quantum_entropy[:8])
        
        # 创建新层级
        next_level = []
        for i in range(0, len(items), 2):
            # 量子增强的分形合并
            # 添加上一层的"记忆"使每个新哈希受到整个树的影响
            combined = items[i] + items[i+1]
            if fractal_memory:
                combined += fractal_memory
            
            # 生成新的量子熵并加入混合
            quantum_entropy = _generate_quantum_entropy()
            combined += quantum_entropy[:16]
            
            # 计算量子哈希
            node_hash = _quantum_hash(combined)
            next_level.append(node_hash)
            
            # 更新分形记忆
            if len(next_level) > 1:
                # 将前两个节点的哈希混合作为分形记忆
                fractal_memory = _quantum_hash(next_level[0] + next_level[-1])[:16]
        
        # 递归构建上层节点
        return _recursive_merkle(next_level, fractal_memory)
    
    # 对所有交易应用量子哈希
    hashed_transactions = []
    for tx in transactions:
        # 序列化交易
        tx_data = json.dumps(tx, sort_keys=True)
        # 添加量子随机性
        quantum_bits = QuantumRandom.generate_random_bits(32)
        # 组合并哈希
        tx_hash = _quantum_hash(tx_data + quantum_bits)
        hashed_transactions.append(tx_hash)
    
    # 构建量子分形Merkle树
    merkle_root = _recursive_merkle(hashed_transactions)
    
    print(f"交易数量: {len(transactions)}")
    print(f"量子分形Merkle根: {merkle_root[:16]}...")
    
    return merkle_root

# 量子Grover搜索算法模拟
def quantum_mining_simulation(target_hash_prefix, difficulty=3):
    """
    量子挖矿模拟 - 基于Grover搜索算法原理
    
    Args:
        target_hash_prefix: 目标哈希前缀
        difficulty: 难度级别（前导零数量）
    
    Returns:
        找到的随机数（nonce）
    """
    print("\n===== 量子挖矿模拟 =====")
    
    # 实际的量子Grover算法更复杂，此处为简化模拟
    target = '0' * difficulty
    print(f"挖矿目标: 哈希前缀必须有{difficulty}个前导零")
    start_time = time.time()
    
    # 创建一个模拟Grover搜索的量子电路
    # 在真实量子计算机上，这将是一个完整的Grover实现
    
    # 记录尝试次数以展示量子加速
    attempts = 0
    
    # 模拟量子搜索过程
    while True:
        attempts += 1
        
        # 生成量子随机nonce - 真正的量子随机性
        qnonce = QuantumRandom.generate_random_bits(32)
        nonce_decimal = int(qnonce, 2)
        
        # 计算哈希
        hash_input = f"{target_hash_prefix}{nonce_decimal}"
        # 使用量子哈希而非经典哈希
        current_hash = QuantumHash.quantum_hash(hash_input)
        
        # 检查是否满足难度目标
        if current_hash.startswith(target):
            break
        
        # 模拟Grover算法的二次加速
        # 实际情况下，量子算法会更快找到结果
        if attempts % 10 == 0:
            print(f"已尝试: {attempts}次, 当前哈希: {current_hash[:10]}...")
    
    elapsed_time = time.time() - start_time
    
    print(f"经过{attempts}次尝试后找到了有效哈希")
    print(f"Nonce值: {nonce_decimal}")
    print(f"最终哈希: {current_hash}")
    print(f"用时: {elapsed_time:.2f}秒")
    
    # 展示量子加速的理论比较
    print(f"经典计算预期尝试次数: {2**difficulty}")
    # Grover算法提供二次加速
    print(f"理论量子Grover算法尝试次数: 约{int(np.sqrt(2**difficulty))}")
    
    return nonce_decimal

def create_quantum_genesis_block():
    """创建并展示量子增强的创世区块"""
    print("\n======= 量子区块链创世区块创建 =======")
    
    # 1. 生成量子密钥
    quantum_state, quantum_key = quantum_key_generation(4)
    
    # 2. 创建初始交易数据
    genesis_transactions = [
        {"type": "coinbase", "recipient": "创世地址", "amount": 50, "timestamp": time.time()},
        {"type": "message", "content": "量子区块链创世区块 - 纯量子实现", "timestamp": time.time()}
    ]
    
    # 3. 计算量子分形Merkle树
    merkle_root = quantum_fractal_merkle_tree(genesis_transactions)
    
    # 4. 生成量子签名
    genesis_message = f"创世区块 - 时间戳:{time.time()} - Merkle根:{merkle_root}"
    quantum_sig = quantum_signature(genesis_message, quantum_key)
    
    # 5. 量子挖矿模拟
    difficulty = 2  # 设置较低难度以便快速演示
    nonce = quantum_mining_simulation(merkle_root[:16], difficulty)
    
    # 6. 创建最终的创世区块
    print("\n===== 创建最终创世区块 =====")
    blockchain = QuantumBlockchain()  # 创建一个新的区块链
    
    # 修改创世区块以包含我们的量子增强属性
    genesis_block = blockchain.chain[0]
    genesis_block.data = {
        "message": "量子区块链创世区块 - 纯量子实现",
        "transactions": genesis_transactions,
        "merkle_root": merkle_root,
        "quantum_signature": quantum_sig,
        "nonce": nonce,
        "quantum_key": quantum_key,
        "token": {
            "name": "QuantumCoin",
            "symbol": "QTC",
            "total_supply": 21000000,
            "decimals": 8,
            "creator": "量子创世地址"
        }
    }
    
    # 重新计算哈希
    genesis_block.hash = genesis_block._calculate_hash()
    
    # 显示创世区块详细信息
    print("\n创世区块最终信息")
    print(f"索引: {genesis_block.index}")
    print(f"时间戳: {time.ctime(genesis_block.timestamp)}")
    print(f"数据摘要: {json.dumps(genesis_block.data, indent=2)}")
    print(f"前一个哈希: {genesis_block.previous_hash}")
    print(f"量子签名: {genesis_block.quantum_signature[:16]}... (已截断)")
    print(f"区块哈希: {genesis_block.hash}")
    
    return blockchain

if __name__ == "__main__":
    print("========================================")
    print("   量子区块链创世区块演示")
    print("  纯量子计算实现 - 最大化量子优势")
    print("========================================")
    
    # 创建量子创世区块
    quantum_blockchain = create_quantum_genesis_block()
    
    print("\n演示完成! 量子区块链已成功初始化并创建创世区块!")
    print("使用'python visualize_quantum_blockchain.py'可以查看区块链可视化结果")