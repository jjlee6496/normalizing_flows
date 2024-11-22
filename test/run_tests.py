import unittest
import sys
import os

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """모든 테스트를 실행하는 함수"""
    # tests 디렉토리의 모든 테스트를 찾아서 실행
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 결과를 터미널에 출력
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_all_tests()