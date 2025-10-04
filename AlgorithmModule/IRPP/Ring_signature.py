import hashlib
import secrets
import unittest
import time
import statistics
from typing import List, Tuple, Optional, Dict
import numpy as np


class RingSignature:
    def __init__(self):
        # SM2 curve parameters
        self.p = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
        self.a = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
        self.b = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
        self.n = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
        self.Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
        self.Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0

        # Generator point G
        self.G = (self.Gx, self.Gy)

    def _point_add(self, P1: Optional[Tuple[int, int]], P2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Elliptic curve point addition"""
        if P1 is None:
            return P2
        if P2 is None:
            return P1

        x1, y1 = P1
        x2, y2 = P2

        if x1 == x2:
            if y1 == y2:
                # Point doubling
                s = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.p) % self.p
            else:
                return None  # Point at infinity
        else:
            s = (y2 - y1) * pow(x2 - x1, -1, self.p) % self.p

        x3 = (s * s - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p

        return (x3, y3)

    def _point_mult(self, k: int, P: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Scalar multiplication of elliptic curve point"""
        if k == 0 or P is None:
            return None
        if k == 1:
            return P

        result = None
        addend = P

        while k:
            if k & 1:
                result = self._point_add(result, addend)
            addend = self._point_add(addend, addend)
            k >>= 1

        return result

    def _sm3_hash(self, data: bytes) -> bytes:
        """SM3 hash function implementation using SHA-256 as substitute"""
        return hashlib.sha256(data).digest()

    def _hash_to_point(self, data: bytes) -> Tuple[int, int]:
        """Hash function H_p: maps data to a point on the elliptic curve"""
        counter = 0
        while counter < 100:
            hash_input = data + counter.to_bytes(4, 'big')
            hash_val = self._sm3_hash(hash_input)
            x = int.from_bytes(hash_val, 'big') % self.p

            rhs = (pow(x, 3, self.p) + self.a * x + self.b) % self.p

            if pow(rhs, (self.p - 1) // 2, self.p) == 1:
                y = pow(rhs, (self.p + 1) // 4, self.p)
                if (y * y) % self.p == rhs:
                    return (x, y)

            counter += 1

        return self.G

    def _hash_to_zq(self, *args) -> int:
        """Hash function H_1: maps inputs to Z_q*"""
        data = b''
        for arg in args:
            if isinstance(arg, bytes):
                data += arg
            elif isinstance(arg, str):
                data += arg.encode('utf-8')
            elif isinstance(arg, int):
                data += arg.to_bytes(32, 'big')
            elif isinstance(arg, tuple):
                if arg is not None:
                    data += arg[0].to_bytes(32, 'big') + arg[1].to_bytes(32, 'big')
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, tuple) and item is not None:
                        data += item[0].to_bytes(32, 'big') + item[1].to_bytes(32, 'big')

        hash_val = self._sm3_hash(data)
        result = int.from_bytes(hash_val, 'big') % self.n
        return result if result != 0 else 1

    def generate_keypair(self) -> Tuple[int, Tuple[int, int]]:
        """Generate a private/public keypair"""
        private_key = secrets.randbelow(self.n - 1) + 1
        public_key = self._point_mult(private_key, self.G)
        return private_key, public_key

    def sign(self, message: str, signer_private_key: int, signer_index: int,
             public_keys: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], int, List[int]]:
        """Generate ring signature"""
        n = len(public_keys)
        π = signer_index

        L = public_keys
        L_bytes = b''.join([pk[0].to_bytes(32, 'big') + pk[1].to_bytes(32, 'big') for pk in L])
        R = self._hash_to_point(L_bytes)

        Q_π = self._point_mult(signer_private_key, R)
        φ_π = secrets.randbelow(self.n - 1) + 1

        φ_π_G = self._point_mult(φ_π, self.G)
        φ_π_R = self._point_mult(φ_π, R)
        c_next = self._hash_to_zq(L, Q_π, message, φ_π_G, φ_π_R)

        c = [0] * (n + 1)
        s = [0] * n
        c[(π + 1) % n] = c_next

        for i in range(n - 1):
            idx = (π + 1 + i) % n
            if idx == π:
                break

            s[idx] = secrets.randbelow(self.n - 1) + 1

            s_i_G = self._point_mult(s[idx], self.G)
            s_i_plus_c_i = (s[idx] + c[idx]) % self.n
            s_i_plus_c_i_K_i = self._point_mult(s_i_plus_c_i, public_keys[idx])
            V_i = self._point_add(s_i_G, s_i_plus_c_i_K_i)

            s_i_R = self._point_mult(s[idx], R)
            s_i_plus_c_i_Q_π = self._point_mult(s_i_plus_c_i, Q_π)
            W_i = self._point_add(s_i_R, s_i_plus_c_i_Q_π)

            c[(idx + 1) % n] = self._hash_to_zq(L, Q_π, message, V_i, W_i)

        k_π = signer_private_key
        c_π = c[π]

        inv_1_plus_k_π = pow(1 + k_π, -1, self.n)
        s[π] = (inv_1_plus_k_π * (φ_π - c_π * k_π)) % self.n

        return (Q_π, c[0], s)

    def verify(self, message: str, signature: Tuple[Tuple[int, int], int, List[int]],
               public_keys: List[Tuple[int, int]]) -> bool:
        """Verify ring signature"""
        try:
            Q_π_prime, c_1_prime, s_prime = signature
            n = len(public_keys)

            L = public_keys
            L_bytes = b''.join([pk[0].to_bytes(32, 'big') + pk[1].to_bytes(32, 'big') for pk in L])
            R = self._hash_to_point(L_bytes)

            if not (1 <= c_1_prime < self.n):
                return False
            for s_i in s_prime:
                if not (1 <= s_i < self.n):
                    return False

            c = [0] * (n + 1)
            c[0] = c_1_prime

            for i in range(n):
                s_i_G = self._point_mult(s_prime[i], self.G)
                s_i_plus_c_i = (s_prime[i] + c[i]) % self.n
                s_i_plus_c_i_K_i = self._point_mult(s_i_plus_c_i, public_keys[i])
                V_i_prime = self._point_add(s_i_G, s_i_plus_c_i_K_i)

                s_i_R = self._point_mult(s_prime[i], R)
                s_i_plus_c_i_Q_π = self._point_mult(s_i_plus_c_i, Q_π_prime)
                W_i_prime = self._point_add(s_i_R, s_i_plus_c_i_Q_π)

                c[i + 1] = self._hash_to_zq(L, Q_π_prime, message, V_i_prime, W_i_prime)

            return c_1_prime == c[n]

        except Exception as e:
            print(f"Verification error: {e}")
            return False

    def are_linkable(self, sig1: Tuple[Tuple[int, int], int, List[int]],
                     sig2: Tuple[Tuple[int, int], int, List[int]],
                     public_keys1: List[Tuple[int, int]],
                     public_keys2: List[Tuple[int, int]],
                     message1: str, message2: str) -> bool:
        """Check if two signatures are linkable"""
        if not (self.verify(message1, sig1, public_keys1) and
                self.verify(message2, sig2, public_keys2)):
            return False

        Q_π1, _, _ = sig1
        Q_π2, _, _ = sig2

        return Q_π1 == Q_π2


class RingSignaturePerformanceTest:
    """Ring Signature performance testing class"""

    def __init__(self, ring_signature: RingSignature):
        self.rs = ring_signature
        self.results = {
            'total_times': [],
            'keygen_times': [],
            'sign_times': [],
            'verify_times': [],
            'linkability_times': []
        }

    def run_performance_test(self, num_iterations: int = 100, ring_size: int = 5,
                             report_interval: int = 10) -> Dict:
        """Run comprehensive performance test"""
        print(f"Starting Ring Signature Performance Test")
        print(f"Iterations: {num_iterations}, Ring Size: {ring_size}")
        print("=" * 60)

        print("Pre-generating keypairs...")
        keypairs, public_keys = self._generate_test_keypairs(ring_size)

        total_start_time = time.time()
        cumulative_times = []

        for i in range(num_iterations):
            iteration_start = time.time()

            message = f"Performance test message #{i + 1}"
            signer_index = i % ring_size
            signer_private_key = keypairs[signer_index][0]

            # Measure signing time
            sign_start = time.time()
            signature = self.rs.sign(message, signer_private_key, signer_index, public_keys)
            sign_time = time.time() - sign_start
            self.results['sign_times'].append(sign_time)

            # Measure verification time
            verify_start = time.time()
            is_valid = self.rs.verify(message, signature, public_keys)
            verify_time = time.time() - verify_start
            self.results['verify_times'].append(verify_time)

            assert is_valid, f"Signature verification failed at iteration {i + 1}"

            # Test linkability every 5 iterations
            if i > 0 and i % 5 == 0:
                linkability_start = time.time()
                prev_message = f"Performance test message #{i}"
                prev_signature = self.rs.sign(prev_message, signer_private_key, signer_index, public_keys)
                are_linked = self.rs.are_linkable(
                    signature, prev_signature,
                    public_keys, public_keys,
                    message, prev_message
                )
                linkability_time = time.time() - linkability_start
                self.results['linkability_times'].append(linkability_time)

                assert are_linked, f"Linkability test failed at iteration {i + 1}"

            iteration_time = time.time() - iteration_start
            self.results['total_times'].append(iteration_time)

            # Report progress
            if (i + 1) % report_interval == 0:
                cumulative_time = time.time() - total_start_time
                cumulative_times.append(cumulative_time)
                avg_time = cumulative_time / (i + 1)

                # === 新增记录功能：每10次操作的累计执行时间 ===
                self._print_detailed_interval_report(i + 1, report_interval, cumulative_time)

        total_time = time.time() - total_start_time

        stats = self._calculate_statistics(total_time, num_iterations)
        self._print_final_results(stats, cumulative_times, report_interval)

        return stats

    # === 详细的间隔报告 ===
    def _print_detailed_interval_report(self, current_iteration: int, interval: int, cumulative_time: float):
        """Print detailed interval report with cumulative times for each operation"""
        start_idx = current_iteration - interval
        end_idx = current_iteration

        # 计算这10次操作的累计时间
        keygen_cumulative = sum(self.results['keygen_times']) if self.results['keygen_times'] else 0
        sign_cumulative = sum(self.results['sign_times'][start_idx:end_idx])
        verify_cumulative = sum(self.results['verify_times'][start_idx:end_idx])

        # 计算可链接性检查的累计时间（只统计这个区间内的）
        linkability_in_interval = []
        linkability_iterations = []
        for i in range(start_idx, end_idx):
            if i > 0 and i % 5 == 0:  # 根据原逻辑，每5次进行可链接性检查
                linkability_iterations.append(i)

        # 找到对应的可链接性检查时间
        linkability_cumulative = 0
        linkability_count = 0
        for i, iter_num in enumerate(linkability_iterations):
            if i < len(self.results['linkability_times']):
                linkability_cumulative += self.results['linkability_times'][-(len(linkability_iterations) - i)]
                linkability_count += 1

        print(f"Completed {current_iteration:3d}/{100} iterations")
        print(f"  Total cumulative time: {cumulative_time:.3f}s")
        print(f"  Average per iteration: {cumulative_time / current_iteration:.4f}s")
        print()
        print(f"  Last {interval} iterations breakdown:")
        print(f"    Key Generation  (total): {keygen_cumulative:.4f}s")
        print(f"    Signing         (累计): {sign_cumulative:.4f}s")
        print(f"    Verification    (累计): {verify_cumulative:.4f}s")
        print(f"    Linkability     (累计): {linkability_cumulative:.4f}s ({linkability_count} checks)")
        print(
            f"    Other overhead  (估计): {cumulative_time - keygen_cumulative - sum(self.results['sign_times']) - sum(self.results['verify_times']) - sum(self.results['linkability_times']):.4f}s")
        print()
        print(f"  Recent {interval} iterations averages:")
        print(f"    Signing avg:      {sign_cumulative / interval:.4f}s")
        print(f"    Verification avg: {verify_cumulative / interval:.4f}s")
        if linkability_count > 0:
            print(f"    Linkability avg:  {linkability_cumulative / linkability_count:.4f}s")
        print("-" * 60)

    def _generate_test_keypairs(self, ring_size: int):
        """Generate keypairs for testing"""
        keypairs = []
        public_keys = []

        keygen_times = []
        for i in range(ring_size):
            start_time = time.time()
            private_key, public_key = self.rs.generate_keypair()
            keygen_time = time.time() - start_time

            keypairs.append((private_key, public_key))
            public_keys.append(public_key)
            keygen_times.append(keygen_time)

        self.results['keygen_times'] = keygen_times
        print(f"Generated {ring_size} keypairs in {sum(keygen_times):.4f}s")
        print(f"Average keygen time: {np.mean(keygen_times):.4f}s")
        print()

        return keypairs, public_keys

    def _calculate_statistics(self, total_time: float, num_iterations: int) -> Dict:
        """Calculate comprehensive statistics"""
        return {
            'total_time': total_time,
            'num_iterations': num_iterations,
            'avg_total_per_iteration': total_time / num_iterations,

            'keygen_stats': {
                'mean': statistics.mean(self.results['keygen_times']),
                'median': statistics.median(self.results['keygen_times']),
                'stdev': statistics.stdev(self.results['keygen_times']) if len(self.results['keygen_times']) > 1 else 0,
                'min': min(self.results['keygen_times']),
                'max': max(self.results['keygen_times'])
            },

            'sign_stats': {
                'mean': statistics.mean(self.results['sign_times']),
                'median': statistics.median(self.results['sign_times']),
                'stdev': statistics.stdev(self.results['sign_times']),
                'min': min(self.results['sign_times']),
                'max': max(self.results['sign_times']),
                'total': sum(self.results['sign_times'])
            },

            'verify_stats': {
                'mean': statistics.mean(self.results['verify_times']),
                'median': statistics.median(self.results['verify_times']),
                'stdev': statistics.stdev(self.results['verify_times']),
                'min': min(self.results['verify_times']),
                'max': max(self.results['verify_times']),
                'total': sum(self.results['verify_times'])
            },

            'linkability_stats': {
                'mean': statistics.mean(self.results['linkability_times']) if self.results['linkability_times'] else 0,
                'median': statistics.median(self.results['linkability_times']) if self.results[
                    'linkability_times'] else 0,
                'stdev': statistics.stdev(self.results['linkability_times']) if len(
                    self.results['linkability_times']) > 1 else 0,
                'min': min(self.results['linkability_times']) if self.results['linkability_times'] else 0,
                'max': max(self.results['linkability_times']) if self.results['linkability_times'] else 0,
                'count': len(self.results['linkability_times'])
            },

            'iteration_stats': {
                'mean': statistics.mean(self.results['total_times']),
                'median': statistics.median(self.results['total_times']),
                'stdev': statistics.stdev(self.results['total_times']),
                'min': min(self.results['total_times']),
                'max': max(self.results['total_times'])
            }
        }

    def _print_final_results(self, stats: Dict, cumulative_times: List[float], report_interval: int):
        """Print comprehensive final results"""
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE RESULTS")
        print("=" * 60)

        print(f"Total execution time: {stats['total_time']:.3f}s")
        print(f"Total iterations: {stats['num_iterations']}")
        print(f"Average time per iteration: {stats['avg_total_per_iteration']:.4f}s")
        print()

        # === 详细的每10次累计时间表格 ===
        print("DETAILED CUMULATIVE EXECUTION TIMES BY OPERATION:")
        print("-" * 80)
        print("Iterations | Total(s) | KeyGen(s) | Sign(s)  | Verify(s) | Link(s)  | Overhead(s)")
        print("-" * 80)

        keygen_total = sum(self.results['keygen_times'])
        sign_running = 0
        verify_running = 0
        linkability_running = 0
        linkability_idx = 0

        for i, cum_time in enumerate(cumulative_times):
            start_iter = i * report_interval
            end_iter = (i + 1) * report_interval

            # 累计签名时间
            sign_interval = sum(self.results['sign_times'][start_iter:end_iter])
            sign_running += sign_interval

            # 累计验证时间
            verify_interval = sum(self.results['verify_times'][start_iter:end_iter])
            verify_running += verify_interval

            # 累计可链接性检查时间
            linkability_interval = 0
            for j in range(start_iter, end_iter):
                if j > 0 and j % 5 == 0 and linkability_idx < len(self.results['linkability_times']):
                    linkability_interval += self.results['linkability_times'][linkability_idx]
                    linkability_idx += 1
            linkability_running += linkability_interval

            # 计算其他开销
            overhead = cum_time - keygen_total - sign_running - verify_running - linkability_running

            print(
                f"{end_iter:8d}   | {cum_time:7.3f}  | {keygen_total:8.4f}  | {sign_running:7.4f} | {verify_running:8.4f}  | {linkability_running:7.4f} | {overhead:9.4f}")

        print("-" * 80)
        print()

        # 原有的统计信息
        operations = [
            ('Key Generation', stats['keygen_stats']),
            ('Signing', stats['sign_stats']),
            ('Verification', stats['verify_stats']),
            ('Linkability Check', stats['linkability_stats']),
            ('Total per Iteration', stats['iteration_stats'])
        ]

        print("DETAILED OPERATION STATISTICS:")
        print("-" * 80)
        print(
            f"{'Operation':<20} | {'Mean(s)':<8} | {'Median(s)':<10} | {'Std Dev':<8} | {'Min(s)':<7} | {'Max(s)':<7}")
        print("-" * 80)

        for op_name, op_stats in operations:
            if op_stats['mean'] > 0:
                print(f"{op_name:<20} | {op_stats['mean']:<8.4f} | {op_stats['median']:<10.4f} | "
                      f"{op_stats['stdev']:<8.4f} | {op_stats['min']:<7.4f} | {op_stats['max']:<7.4f}")
        print("-" * 80)
        print()

        # Performance breakdown
        total_sign_time = stats['sign_stats']['total']
        total_verify_time = stats['verify_stats']['total']
        total_ops_time = total_sign_time + total_verify_time

        print("OPERATION TIME BREAKDOWN:")
        print("-" * 40)
        print(f"Total signing time:     {total_sign_time:.3f}s ({total_sign_time / stats['total_time'] * 100:.1f}%)")
        print(
            f"Total verification time: {total_verify_time:.3f}s ({total_verify_time / stats['total_time'] * 100:.1f}%)")
        print(
            f"Other overhead:         {stats['total_time'] - total_ops_time:.3f}s ({(stats['total_time'] - total_ops_time) / stats['total_time'] * 100:.1f}%)")
        print()

        # Throughput metrics
        print("THROUGHPUT METRICS:")
        print("-" * 40)
        print(f"Signatures per second:   {stats['num_iterations'] / stats['total_time']:.2f}")
        print(f"Verifications per second: {stats['num_iterations'] / total_verify_time:.2f}")
        print(f"Full cycles per second:   {stats['num_iterations'] / stats['total_time']:.2f}")


class TestRingSignature(unittest.TestCase):
    """Unit tests for Ring Signature implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.rs = RingSignature()
        self.ring_size = 5

        self.keypairs = []
        self.public_keys = []

        for i in range(self.ring_size):
            private_key, public_key = self.rs.generate_keypair()
            self.keypairs.append((private_key, public_key))
            self.public_keys.append(public_key)

    def test_keypair_generation(self):
        """Test keypair generation"""
        private_key, public_key = self.rs.generate_keypair()

        self.assertGreater(private_key, 0)
        self.assertLess(private_key, self.rs.n)

        self.assertIsInstance(public_key, tuple)
        self.assertEqual(len(public_key), 2)

        expected_public_key = self.rs._point_mult(private_key, self.rs.G)
        self.assertEqual(public_key, expected_public_key)

    def test_signature_generation_and_verification(self):
        """Test signature generation and verification"""
        message = "Test message for ring signature"
        signer_index = 2
        signer_private_key = self.keypairs[signer_index][0]

        signature = self.rs.sign(message, signer_private_key, signer_index, self.public_keys)

        self.assertIsInstance(signature, tuple)
        self.assertEqual(len(signature), 3)
        Q_π, c_1, s_list = signature
        self.assertIsInstance(Q_π, tuple)
        self.assertIsInstance(c_1, int)
        self.assertIsInstance(s_list, list)
        self.assertEqual(len(s_list), self.ring_size)

        is_valid = self.rs.verify(message, signature, self.public_keys)
        self.assertTrue(is_valid)

    def test_linkability_same_signer(self):
        """Test that signatures from same signer are linkable"""
        message1 = "First message"
        message2 = "Second message"
        signer_index = 3
        signer_private_key = self.keypairs[signer_index][0]

        signature1 = self.rs.sign(message1, signer_private_key, signer_index, self.public_keys)
        signature2 = self.rs.sign(message2, signer_private_key, signer_index, self.public_keys)

        are_linked = self.rs.are_linkable(
            signature1, signature2,
            self.public_keys, self.public_keys,
            message1, message2
        )
        self.assertTrue(are_linked)

    def test_performance_100_iterations(self):
        """Test 100 iterations of ring signature operations"""
        print("\n" + "=" * 60)
        print("RUNNING 100-ITERATION PERFORMANCE TEST")
        print("=" * 60)

        perf_tester = RingSignaturePerformanceTest(self.rs)
        stats = perf_tester.run_performance_test(
            num_iterations=100,
            ring_size=5,
            report_interval=10
        )

        self.assertLess(stats['avg_total_per_iteration'], 2.0, "Average iteration time should be reasonable")
        self.assertLess(stats['sign_stats']['mean'], 1.0, "Average signing time should be reasonable")
        self.assertLess(stats['verify_stats']['mean'], 1.0, "Average verification time should be reasonable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
