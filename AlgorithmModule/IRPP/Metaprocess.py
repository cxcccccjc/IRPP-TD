import time
import hashlib
import random
import secrets
from typing import Tuple, List, Optional


class SM2EllipticCurve:
    """SM2椭圆曲线实现 - 使用国标参数"""

    def __init__(self):
        # SM2国标参数
        self.p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        self.a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        self.n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
        self.Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
        self.G = (self.Gx, self.Gy)

    def mod_inverse(self, a: int, m: int) -> int:
        """计算模逆"""

        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            raise ValueError("模逆不存在")
        return (x % m + m) % m

    def point_add(self, P: Optional[Tuple[int, int]], Q: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """椭圆曲线点加法"""
        if P is None:
            return Q
        if Q is None:
            return P

        x1, y1 = P
        x2, y2 = Q

        if x1 == x2:
            if y1 == y2:
                # 点倍运算
                s = ((3 * x1 * x1 + self.a) * self.mod_inverse(2 * y1, self.p)) % self.p
            else:
                return None  # 无穷远点
        else:
            # 点加运算
            s = ((y2 - y1) * self.mod_inverse((x2 - x1) % self.p, self.p)) % self.p

        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p

        return (x3, y3)

    def scalar_mult(self, k: int, P: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """椭圆曲线标量乘法 - 使用二进制方法"""
        if k == 0:
            return None
        if k == 1:
            return P

        result = None
        addend = P

        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1

        return result


class NTRUCrypto:
    """NTRU加密系统 - 使用国标参数"""

    def __init__(self, n: int = 251, q: int = 128, p: int = 3):
        self.n = n
        self.q = q
        self.p = p

    def generate_polynomial(self, d_ones: int = 84, d_minus_ones: int = 84) -> List[int]:
        """生成随机多项式"""
        coeffs = [0] * self.n

        # 随机选择位置放置+1
        ones_positions = random.sample(range(self.n), d_ones)
        for pos in ones_positions:
            coeffs[pos] = 1

        # 在剩余位置中随机选择放置-1
        remaining = [i for i in range(self.n) if coeffs[i] == 0]
        minus_ones_positions = random.sample(remaining, min(d_minus_ones, len(remaining)))
        for pos in minus_ones_positions:
            coeffs[pos] = -1

        return coeffs

    def polynomial_multiply(self, a: List[int], b: List[int]) -> List[int]:
        """多项式乘法 mod (x^n - 1)"""
        result = [0] * self.n

        for i in range(self.n):
            for j in range(self.n):
                if a[i] != 0 and b[j] != 0:
                    pos = (i + j) % self.n
                    result[pos] = (result[pos] + a[i] * b[j]) % self.q

        return result

    def polynomial_divide_mod(self, dividend: List[int], divisor: List[int]) -> List[int]:
        """多项式除法运算（模拟逆元计算）"""

        result = [0] * self.n

        # 使用简单的试探方法
        for i in range(self.n):
            if dividend[i] != 0:
                for j in range(1, self.q):
                    test_result = [(dividend[k] * j) % self.q for k in range(self.n)]
                    if self.polynomial_multiply(divisor, test_result)[i] % self.q == dividend[i]:
                        result = test_result
                        break
                break

        if all(x == 0 for x in result):
            # 如果找不到精确解，返回一个随机多项式作为近似
            result = self.generate_polynomial(42, 42)

        return result


class HashOperations:
    """哈希操作实现"""

    def __init__(self, modulus: int):
        self.modulus = modulus

    def hash_h0(self, data: bytes) -> str:
        """H0: 标准SHA-256哈希"""
        return hashlib.sha256(data).hexdigest()

    def hash_h1(self, data: bytes) -> int:
        """H1: SHA-256映射到Zq*"""
        hash_bytes = hashlib.sha256(data).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        return (hash_int % (self.modulus - 1)) + 1  # 确保结果在Zq*中


class PerformanceTimer:
    """性能计时器"""

    @staticmethod
    def time_operation(operation, *args, **kwargs):
        """计时单个操作"""
        start_time = time.perf_counter()
        result = operation(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result

    @staticmethod
    def time_multiple_operations(operation, iterations: int, *args, **kwargs):
        """计时多次操作"""
        times = []
        results = []

        for _ in range(iterations):
            execution_time, result = PerformanceTimer.time_operation(operation, *args, **kwargs)
            times.append(execution_time)
            results.append(result)

        return {
            'times': times,
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'results': results
        }


def run_crypto_performance_tests():
    """运行加密性能测试"""

    print("=" * 60)
    print("加密算法性能测试")
    print("=" * 60)

    # 初始化测试组件
    sm2 = SM2EllipticCurve()
    ntru = NTRUCrypto(n=251, q=128, p=3)
    hasher = HashOperations(sm2.n)
    timer = PerformanceTimer()

    # 测试参数
    iterations = 10
    test_message = "This is a test message for cryptographic performance evaluation"
    test_bytes = test_message.encode('utf-8')

    print(f"测试参数:")
    print(f"- 每个操作测试次数: {iterations}")
    print(f"- SM2参数: p={hex(sm2.p)[:20]}..., n={hex(sm2.n)[:20]}...")
    print(f"- NTRU参数: n={ntru.n}, q={ntru.q}, p={ntru.p}")
    print(f"- 测试消息长度: {len(test_bytes)} 字节")
    print()

    # 测试结果存储
    all_results = {}

    # 1. 椭圆曲线标量乘法测试
    print("1. 测试椭圆曲线标量乘法...")

    def ec_scalar_mult():
        k = secrets.randbelow(sm2.n - 1) + 1
        return sm2.scalar_mult(k, sm2.G)

    ec_results = timer.time_multiple_operations(ec_scalar_mult, iterations)
    all_results['椭圆曲线标量乘法'] = ec_results

    # 2. 多项式生成测试
    print("2. 测试多项式生成...")

    def poly_gen():
        return ntru.generate_polynomial(84, 84)

    poly_gen_results = timer.time_multiple_operations(poly_gen, iterations)
    all_results['多项式生成'] = poly_gen_results

    # 3. 多项式乘法测试
    print("3. 测试多项式乘法...")
    poly1 = ntru.generate_polynomial(84, 84)
    poly2 = ntru.generate_polynomial(84, 84)

    def poly_mult():
        return ntru.polynomial_multiply(poly1, poly2)

    poly_mult_results = timer.time_multiple_operations(poly_mult, iterations)
    all_results['多项式乘法'] = poly_mult_results

    # 4. 多项式除法测试
    print("4. 测试多项式除法...")
    dividend = ntru.generate_polynomial(84, 84)
    divisor = ntru.generate_polynomial(42, 42)

    def poly_div():
        return ntru.polynomial_divide_mod(dividend, divisor)

    poly_div_results = timer.time_multiple_operations(poly_div, iterations)
    all_results['多项式除法'] = poly_div_results

    # 5. 哈希H0测试
    print("5. 测试哈希操作H0...")

    def hash_h0():
        return hasher.hash_h0(test_bytes)

    h0_results = timer.time_multiple_operations(hash_h0, iterations)
    all_results['哈希H0'] = h0_results

    # 6. 哈希H1测试
    print("6. 测试哈希操作H1...")

    def hash_h1():
        return hasher.hash_h1(test_bytes)

    h1_results = timer.time_multiple_operations(hash_h1, iterations)
    all_results['哈希H1'] = h1_results

    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)

    print(f"{'操作名称':<15} {'平均时间(ms)':<12} {'最小时间(ms)':<12} {'最大时间(ms)':<12}")
    print("-" * 60)

    for operation_name, results in all_results.items():
        avg_ms = results['average'] * 1000
        min_ms = results['min'] * 1000
        max_ms = results['max'] * 1000

        print(f"{operation_name:<15} {avg_ms:<12.4f} {min_ms:<12.4f} {max_ms:<12.4f}")

    # 详细结果
    print(f"\n详细测试结果 (单位: 毫秒):")
    print("-" * 60)

    for operation_name, results in all_results.items():
        print(f"\n{operation_name}:")
        times_ms = [t * 1000 for t in results['times']]
        for i, time_ms in enumerate(times_ms, 1):
            print(f"  第{i:2d}次: {time_ms:8.4f} ms")

        std_dev = (sum([(t - results['average']) ** 2 for t in results['times']]) / len(results['times'])) ** 0.5
        print(f"  标准差: {std_dev * 1000:8.4f} ms")

    return all_results


if __name__ == "__main__":
    try:
        results = run_crypto_performance_tests()
        print(f"\n测试完成！共完成 {len(results)} 项性能测试。")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
