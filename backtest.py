import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class BitcoinPairTradingBacktest:
    def __init__(self, symbol1='BTC-USD', symbol2='ETH-USD'):
        """
        비트코인 페어 트레이딩 백테스트 클래스
        
        Parameters:
        symbol1: 첫 번째 종목 (예: 'BTC-USD')
        symbol2: 두 번째 종목 (예: 'ETH-USD') 
        """
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.data = None
        self.results = None
        
    def fetch_data(self, start_date=None, end_date=None, period='1y'):
        """
        데이터 다운로드
        
        Parameters:
        start_date: 시작일 ('YYYY-MM-DD' 형식)
        end_date: 종료일 ('YYYY-MM-DD' 형식)
        period: 기간 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        print(f"{self.symbol1}과 {self.symbol2} 데이터를 다운로드 중...")
        
        if start_date and end_date:
            data1 = yf.download(self.symbol1, start=start_date, end=end_date, progress=False)
            data2 = yf.download(self.symbol2, start=start_date, end=end_date, progress=False)
        else:
            data1 = yf.download(self.symbol1, period=period, progress=False)
            data2 = yf.download(self.symbol2, period=period, progress=False)
        
        # 데이터 정리
        df = pd.DataFrame({
            f'{self.symbol1}_close': data1['Close'],
            f'{self.symbol2}_close': data2['Close']
        }).dropna()
        
        # 가격 비율 계산
        df['price_ratio'] = df[f'{self.symbol1}_close'] / df[f'{self.symbol2}_close']
        
        self.data = df
        print(f"데이터 다운로드 완료: {len(df)}개 데이터포인트")
        return df
    
    def calculate_signals(self, lookback_window=20, entry_threshold=2.0, exit_threshold=0.5):
        """
        트레이딩 신호 계산
        
        Parameters:
        lookback_window: 이동평균 및 표준편차 계산 윈도우
        entry_threshold: 진입 신호 임계값 (표준편차 배수)
        exit_threshold: 청산 신호 임계값 (표준편차 배수)
        """
        if self.data is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        df = self.data.copy()
        
        # 이동평균과 표준편차 계산
        df['ratio_ma'] = df['price_ratio'].rolling(window=lookback_window).mean()
        df['ratio_std'] = df['price_ratio'].rolling(window=lookback_window).std()
        
        # Z-score 계산
        df['z_score'] = (df['price_ratio'] - df['ratio_ma']) / df['ratio_std']
        
        # 신호 생성
        df['signal'] = 0
        df['position'] = 0
        
        position = 0  # 0: 중립, 1: 롱 포지션, -1: 숏 포지션
        
        for i in range(lookback_window, len(df)):
            z = df['z_score'].iloc[i]
            
            if position == 0:  # 포지션이 없을 때
                if z > entry_threshold:
                    # 비율이 높으면 symbol1 숏, symbol2 롱
                    position = -1
                    df.loc[df.index[i], 'signal'] = -1
                elif z < -entry_threshold:
                    # 비율이 낮으면 symbol1 롱, symbol2 숏
                    position = 1
                    df.loc[df.index[i], 'signal'] = 1
            
            elif position == 1:  # 롱 포지션일 때
                if z > exit_threshold:
                    # 청산
                    position = 0
                    df.loc[df.index[i], 'signal'] = 0
            
            elif position == -1:  # 숏 포지션일 때
                if z < -exit_threshold:
                    # 청산
                    position = 0
                    df.loc[df.index[i], 'signal'] = 0
            
            df.loc[df.index[i], 'position'] = position
        
        self.data = df
        return df
    
    def calculate_returns(self, initial_capital=100000, transaction_cost=0.001):
        """
        수익률 계산
        
        Parameters:
        initial_capital: 초기 자본
        transaction_cost: 거래비용 (0.1% = 0.001)
        """
        if self.data is None or 'position' not in self.data.columns:
            raise ValueError("먼저 calculate_signals()를 실행해주세요.")
        
        df = self.data.copy()
        
        # 일일 수익률 계산
        df[f'{self.symbol1}_returns'] = df[f'{self.symbol1}_close'].pct_change()
        df[f'{self.symbol2}_returns'] = df[f'{self.symbol2}_close'].pct_change()
        
        # 포지션 변화 감지
        df['position_change'] = df['position'].diff().fillna(0)
        
        # 페어 트레이딩 수익률 계산
        df['strategy_returns'] = 0.0
        df['transaction_costs'] = 0.0
        
        for i in range(1, len(df)):
            pos = df['position'].iloc[i-1]  # 전일 포지션
            
            if pos == 1:  # symbol1 롱, symbol2 숏
                daily_return = (df[f'{self.symbol1}_returns'].iloc[i] - 
                               df[f'{self.symbol2}_returns'].iloc[i])
            elif pos == -1:  # symbol1 숏, symbol2 롱
                daily_return = (-df[f'{self.symbol1}_returns'].iloc[i] + 
                               df[f'{self.symbol2}_returns'].iloc[i])
            else:
                daily_return = 0
            
            df.loc[df.index[i], 'strategy_returns'] = daily_return
            
            # 거래비용 계산
            if abs(df['position_change'].iloc[i]) > 0:
                df.loc[df.index[i], 'transaction_costs'] = transaction_cost * 2  # 양방향 거래
        
        # 순수익률 (거래비용 차감)
        df['net_returns'] = df['strategy_returns'] - df['transaction_costs']
        
        # 누적 수익률
        df['cumulative_returns'] = (1 + df['net_returns']).cumprod()
        df['cumulative_value'] = df['cumulative_returns'] * initial_capital
        
        # 벤치마크 (동일가중 포트폴리오)
        df['benchmark_returns'] = (df[f'{self.symbol1}_returns'] + df[f'{self.symbol2}_returns']) / 2
        df['benchmark_cumulative'] = (1 + df['benchmark_returns'].fillna(0)).cumprod() * initial_capital
        
        self.data = df
        return df
    
    def calculate_performance_metrics(self):
        """성과 지표 계산"""
        if self.data is None or 'net_returns' not in self.data.columns:
            raise ValueError("먼저 calculate_returns()를 실행해주세요.")
        
        returns = self.data['net_returns'].dropna()
        benchmark_returns = self.data['benchmark_returns'].dropna()
        
        # 기본 지표
        total_return = (self.data['cumulative_returns'].iloc[-1] - 1) * 100
        benchmark_total_return = (self.data['benchmark_cumulative'].iloc[-1] / self.data['benchmark_cumulative'].iloc[0] - 1) * 100
        
        # 연간화 수익률
        trading_days = len(returns)
        annual_return = (self.data['cumulative_returns'].iloc[-1] ** (252/trading_days) - 1) * 100
        
        # 변동성
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 샤프 비율 (무위험수익률 0% 가정)
        sharpe_ratio = (annual_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # 최대 낙폭
        cumulative = self.data['cumulative_returns']
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # 승률
        winning_trades = len(returns[returns > 0])
        total_trades = len(returns[returns != 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 거래 횟수
        trades = len(self.data[self.data['position_change'] != 0])
        
        metrics = {
            '총 수익률 (%)': round(total_return, 2),
            '연간화 수익률 (%)': round(annual_return, 2),
            '벤치마크 수익률 (%)': round(benchmark_total_return, 2),
            '변동성 (%)': round(volatility, 2),
            '샤프 비율': round(sharpe_ratio, 2),
            '최대 낙폭 (%)': round(max_drawdown, 2),
            '승률 (%)': round(win_rate, 2),
            '총 거래 횟수': trades,
            '거래일 수': trading_days
        }
        
        self.results = metrics
        return metrics
    
    def plot_results(self, figsize=(15, 12)):
        """결과 시각화"""
        if self.data is None:
            raise ValueError("먼저 백테스트를 실행해주세요.")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 1. 가격 추이
        axes[0].plot(self.data.index, self.data[f'{self.symbol1}_close'], 
                    label=self.symbol1, alpha=0.7)
        axes[0].plot(self.data.index, self.data[f'{self.symbol2}_close'], 
                    label=self.symbol2, alpha=0.7)
        axes[0].set_title('가격 추이')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 가격 비율과 Z-score
        ax2_1 = axes[1]
        ax2_2 = ax2_1.twinx()
        
        ax2_1.plot(self.data.index, self.data['price_ratio'], 'b-', alpha=0.7, label='Price Ratio')
        ax2_1.plot(self.data.index, self.data['ratio_ma'], 'r--', alpha=0.7, label='Moving Average')
        ax2_2.plot(self.data.index, self.data['z_score'], 'g-', alpha=0.7, label='Z-score')
        
        ax2_2.axhline(y=2, color='r', linestyle='--', alpha=0.5)
        ax2_2.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax2_2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        ax2_2.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        
        ax2_1.set_ylabel('Price Ratio', color='b')
        ax2_2.set_ylabel('Z-score', color='g')
        ax2_1.set_title('가격 비율과 Z-score')
        ax2_1.grid(True, alpha=0.3)
        
        # 3. 포지션
        axes[2].plot(self.data.index, self.data['position'], drawstyle='steps-post')
        axes[2].set_title('포지션 (1: 롱, -1: 숏, 0: 중립)')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. 누적 수익률
        axes[3].plot(self.data.index, self.data['cumulative_value'], 
                    label='페어 트레이딩 전략', linewidth=2)
        axes[3].plot(self.data.index, self.data['benchmark_cumulative'], 
                    label='벤치마크 (동일가중)', alpha=0.7)
        axes[3].set_title('누적 수익률')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_backtest(self, start_date=None, end_date=None, period='1y', 
                    lookback_window=20, entry_threshold=2.0, exit_threshold=0.5,
                    initial_capital=100000, transaction_cost=0.001):
        """
        전체 백테스트 실행
        
        Parameters:
        start_date: 시작일 ('YYYY-MM-DD')
        end_date: 종료일 ('YYYY-MM-DD')
        period: 기간 ('1y', '6mo', '3mo' 등)
        lookback_window: 이동평균 윈도우
        entry_threshold: 진입 임계값
        exit_threshold: 청산 임계값
        initial_capital: 초기 자본
        transaction_cost: 거래비용
        """
        print("=== 비트코인 페어 트레이딩 백테스트 ===\n")
        
        # 데이터 다운로드
        self.fetch_data(start_date, end_date, period)
        
        # 신호 계산
        print("트레이딩 신호 계산 중...")
        self.calculate_signals(lookback_window, entry_threshold, exit_threshold)
        
        # 수익률 계산
        print("수익률 계산 중...")
        self.calculate_returns(initial_capital, transaction_cost)
        
        # 성과 지표 계산
        metrics = self.calculate_performance_metrics()
        
        # 결과 출력
        print("\n=== 백테스트 결과 ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        return metrics

# 사용 예시
if __name__ == "__main__":
    # 백테스트 실행
    bt = BitcoinPairTradingBacktest('BTC-USD', 'ETH-USD')
    
    # 최근 1년 데이터로 백테스트
    results = bt.run_backtest(
        period='1y',
        lookback_window=20,
        entry_threshold=2.0,
        exit_threshold=0.5,
        initial_capital=100000,
        transaction_cost=0.001
    )
    
    # 결과 시각화
    bt.plot_results()
    
    # 다른 기간으로 테스트 (특정 날짜 지정)
    print("\n" + "="*50)
    print("다른 기간 테스트 예시")
    
    bt2 = BitcoinPairTradingBacktest('BTC-USD', 'ETH-USD')
    results2 = bt2.run_backtest(
        start_date='2023-01-01',
        end_date='2023-12-31',
        lookback_window=15,
        entry_threshold=1.5,
        exit_threshold=0.3
    )
    
    # 여러 파라미터 조합 테스트
    print("\n" + "="*50)
    print("파라미터 최적화 예시")
    
    best_sharpe = -999
    best_params = {}
    
    for window in [10, 20, 30]:
        for entry_th in [1.5, 2.0, 2.5]:
            for exit_th in [0.3, 0.5, 0.7]:
                bt_test = BitcoinPairTradingBacktest('BTC-USD', 'ETH-USD')
                try:
                    bt_test.fetch_data(period='6mo')
                    bt_test.calculate_signals(window, entry_th, exit_th)
                    bt_test.calculate_returns()
                    metrics = bt_test.calculate_performance_metrics()
                    
                    if metrics['샤프 비율'] > best_sharpe:
                        best_sharpe = metrics['샤프 비율']
                        best_params = {
                            'window': window,
                            'entry_threshold': entry_th,
                            'exit_threshold': exit_th,
                            'total_return': metrics['총 수익률 (%)'],
                            'sharpe': metrics['샤프 비율']
                        }
                except:
                    continue
    
    print(f"최적 파라미터: {best_params}")