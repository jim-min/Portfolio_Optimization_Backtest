import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import dotenv
from scipy.optimize import minimize
dotenv.load_dotenv()

warnings.filterwarnings('ignore')
class PortfolioOptimizer:
    def __init__(self, symbol1='ETH', symbol2='DOT', api_key=None):
        """
        포트폴리오 최적화 클래스

        Parameters:
        symbol1: 첫 번째 자산 (예: 'ETH-USD')
        symbol2: 두 번째 자산 (예: 'DOT-USD')
        api_key: Alpha Vantage API Key (문자열)
        """
        self.symbol1, self.market1 = (symbol1.split('-') + ['USD'])[:2]
        self.symbol2, self.market2 = (symbol2.split('-') + ['USD'])[:2]

        if self.market1 != self.market2:
            raise ValueError("두 자산의 마켓(market)이 동일해야 합니다.")
        self.market = self.market1
        
        self.data = None
        self.returns = None
        self.results = {}
        self.api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not self.api_key:
            raise ValueError('Alpha Vantage API key must be provided!')
        self.cc = CryptoCurrencies(key=self.api_key, output_format='pandas')

    def fetch_data(self):
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        data_file = f'data/data_{self.symbol1}_{self.symbol2}_{self.market}.pkl'
        
        # Check if data file exists
        if os.path.exists(data_file):
            print(f"Loading data from {data_file}...")
            self.data = pd.read_pickle(data_file)
            print("Data loaded successfully!")
        else:
            print(f"{self.symbol1}-{self.market}와 {self.symbol2}-{self.market} 데이터를 다운로드 중...")

            try:
                data1, _ = self.cc.get_digital_currency_daily(symbol=self.symbol1, market=self.market)
                print("--- API Response for symbol1 ---")
                print(data1)
                print("---------------------------------")
                data2, _ = self.cc.get_digital_currency_daily(symbol=self.symbol2, market=self.market)
                print("--- API Response for symbol2 ---")
                print(data2)
                print("---------------------------------")
                
                data1 = data1.sort_index()
                data2 = data2.sort_index()
                
                close_col_name = f'4. close'

                df = pd.DataFrame({
                    f'{self.symbol1}_price': data1[close_col_name],
                    f'{self.symbol2}_price': data2[close_col_name]
                }).dropna()

                # Save to pickle
                df.to_pickle(data_file)
                print(f"Data saved to {data_file}")
                
                self.data = df
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                return
            
        # except Exception as e:
        #     print(f"Alpha Vantage 데이터 다운로드 실패: {e}")
        #     return

        self.returns = pd.DataFrame({
            f'{self.symbol1}_return': self.data[f'{self.symbol1}_price'].pct_change(),
            f'{self.symbol2}_return': self.data[f'{self.symbol2}_price'].pct_change()
        }).dropna()

        print(f"전체 데이터 다운로드 완료: {len(self.data)}개 데이터포인트")

    def _create_sample_data(self):
        print("🔄 샘플 데이터를 생성합니다 (데모용)")
        dates = pd.date_range(start=self.data.index[0], end=self.data.index[-1], freq='D')
        np.random.seed(42)
        eth_returns = np.random.normal(0.001, 0.04, len(dates))
        eth_prices = 2000 * np.cumprod(1 + eth_returns)
        dot_returns = 0.7 * eth_returns + 0.3 * np.random.normal(0.0005, 0.05, len(dates))
        dot_prices = 7 * np.cumprod(1 + dot_returns)
        self.data = pd.DataFrame({
            f'{self.symbol1}_price': eth_prices,
            f'{self.symbol2}_price': dot_prices
        }, index=dates)
        self.returns = pd.DataFrame({
            f'{self.symbol1}_return': eth_returns,
            f'{self.symbol2}_return': dot_returns
        }, index=dates)
        print(f"샘플 데이터 생성 완료: {len(self.data)}개 데이터포인트")

    def calculate_portfolio_metrics(self, weight1, weight2=None):
        """
        포트폴리오 성과 지표 계산
        
        Parameters:
        weight1: 첫 번째 자산 비중 (0~1)
        weight2: 두 번째 자산 비중 (None이면 1-weight1로 자동 계산)
        """
        if weight2 is None:
            weight2 = 1 - weight1
            
        if self.returns is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        # 포트폴리오 수익률 계산
        portfolio_returns = (weight1 * self.returns.iloc[:, 0] + 
                           weight2 * self.returns.iloc[:, 1])
        
        # 성과 지표 계산
        annual_return = portfolio_returns.mean() * 252 * 100  # 연간화 수익률 (%)
        annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100  # 연간화 변동성 (%)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 최대 낙폭 계산
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # VaR 계산 (95% 신뢰수준)
        var_95 = np.percentile(portfolio_returns, 5) * 100
        
        # 상관계수
        correlation = self.returns.iloc[:, 0].corr(self.returns.iloc[:, 1])
        
        return {
            'weight1': weight1,
            'weight2': weight2,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'correlation': correlation,
            'portfolio_returns': portfolio_returns
        }
    
    def efficient_frontier(self, num_portfolios=100):
        """
        효율적 투자선 계산
        
        Parameters:
        num_portfolios: 계산할 포트폴리오 개수
        """
        if self.returns is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        print("효율적 투자선 계산 중...")
        
        # 비중 범위 설정 (0%~100%)
        weights1 = np.linspace(0, 1, num_portfolios)
        
        results = []
        for w1 in weights1:
            metrics = self.calculate_portfolio_metrics(w1)
            results.append(metrics)
        
        self.efficient_frontier_data = pd.DataFrame(results)
        return self.efficient_frontier_data
    
    def find_optimal_portfolios(self):
        """
        다양한 최적화 기준에 따른 최적 포트폴리오 찾기
        """
        if self.returns is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        print("최적 포트폴리오 탐색 중...")
        
        # 평균과 공분산 계산
        mean_returns = self.returns.mean() * 252  # 연간화
        cov_matrix = self.returns.cov() * 252     # 연간화
        
        def portfolio_performance(weights):
            returns = np.sum(mean_returns * weights)
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, std
        
        def negative_sharpe(weights):
            returns, std = portfolio_performance(weights)
            return -(returns / std) if std > 0 else -999
        
        def portfolio_volatility(weights):
            return portfolio_performance(weights)[1]
        
        # 제약 조건
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(2))
        
        # 1. 샤프 비율 최대화
        result_sharpe = minimize(negative_sharpe, [0.5, 0.5], 
                               method='SLSQP', bounds=bounds, constraints=constraints)
        
        # 2. 최소 변동성
        result_min_vol = minimize(portfolio_volatility, [0.5, 0.5],
                                method='SLSQP', bounds=bounds, constraints=constraints)
        
        # 3. 등가중 포트폴리오
        equal_weight = [0.5, 0.5]
        
        # 4. 리스크 패리티 (변동성 기여도 동일)
        def risk_parity_objective(weights):
            portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib
            return np.sum((contrib - portfolio_var/2)**2)
        
        result_risk_parity = minimize(risk_parity_objective, [0.5, 0.5],
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        
        # 결과 저장
        optimal_portfolios = {
            'max_sharpe': {
                'weights': result_sharpe.x,
                'metrics': self.calculate_portfolio_metrics(result_sharpe.x[0], result_sharpe.x[1])
            },
            'min_volatility': {
                'weights': result_min_vol.x,
                'metrics': self.calculate_portfolio_metrics(result_min_vol.x[0], result_min_vol.x[1])
            },
            'equal_weight': {
                'weights': equal_weight,
                'metrics': self.calculate_portfolio_metrics(equal_weight[0], equal_weight[1])
            },
            'risk_parity': {
                'weights': result_risk_parity.x,
                'metrics': self.calculate_portfolio_metrics(result_risk_parity.x[0], result_risk_parity.x[1])
            }
        }
        
        self.results = optimal_portfolios
        return optimal_portfolios
    
    def monte_carlo_optimization(self, num_simulations=10000):
        """
        몬테카를로 시뮬레이션을 통한 최적 포트폴리오 탐색
        
        Parameters:
        num_simulations: 시뮬레이션 횟수
        """
        if self.returns is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        print(f"몬테카를로 시뮬레이션 실행 중... ({num_simulations:,}회)")
        
        # 랜덤 비중 생성
        np.random.seed(42)
        weights1 = np.random.random(num_simulations)
        weights2 = 1 - weights1
        
        results = []
        for i in range(num_simulations):
            metrics = self.calculate_portfolio_metrics(weights1[i], weights2[i])
            results.append(metrics)
        
        mc_results = pd.DataFrame(results)
        
        # 최적 포트폴리오 찾기
        best_sharpe_idx = mc_results['sharpe_ratio'].idxmax()
        best_return_idx = mc_results['annual_return'].idxmax()
        min_vol_idx = mc_results['annual_volatility'].idxmin()
        min_drawdown_idx = mc_results['max_drawdown'].idxmax()  # 가장 작은 낙폭
        
        self.mc_results = mc_results
        self.mc_optimal = {
            'best_sharpe': mc_results.loc[best_sharpe_idx],
            'best_return': mc_results.loc[best_return_idx],
            'min_volatility': mc_results.loc[min_vol_idx],
            'min_drawdown': mc_results.loc[min_drawdown_idx]
        }
        
        return mc_results, self.mc_optimal
    
    def rolling_window_analysis(self, window_months=6, rebalance_freq='M'):
        """
        롤링 윈도우를 통한 시계열 최적화 분석
        
        Parameters:
        window_months: 분석 윈도우 (개월)
        rebalance_freq: 리밸런싱 주기 ('M': 월별, 'Q': 분기별)
        """
        if self.returns is None:
            raise ValueError("먼저 fetch_data()를 실행해주세요.")
        
        print(f"롤링 윈도우 분석 중... (윈도우: {window_months}개월)")
        
        window_days = window_months * 21  # 대략적인 거래일 수
        
        # 리밸런싱 날짜 생성
        if rebalance_freq == 'M':
            rebalance_dates = pd.date_range(start=self.returns.index[window_days], 
                                          end=self.returns.index[-1], freq='MS')
        else:  # 분기별
            rebalance_dates = pd.date_range(start=self.returns.index[window_days], 
                                          end=self.returns.index[-1], freq='QS')
        
        rolling_results = []
        
        for date in rebalance_dates:
            # 윈도우 데이터 추출
            end_idx = self.returns.index.get_loc(date, method='nearest')
            start_idx = max(0, end_idx - window_days)
            
            window_returns = self.returns.iloc[start_idx:end_idx]
            
            if len(window_returns) < 20:  # 최소 데이터 요구사항
                continue
            
            # 해당 윈도우에서 최적 비중 계산
            mean_returns = window_returns.mean() * 252
            cov_matrix = window_returns.cov() * 252
            
            def negative_sharpe(weights):
                returns = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(returns / std) if std > 0 else -999
            
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(2))
            
            try:
                result = minimize(negative_sharpe, [0.5, 0.5], 
                                method='SLSQP', bounds=bounds, constraints=constraints)
                
                rolling_results.append({
                    'date': date,
                    'weight1': result.x[0],
                    'weight2': result.x[1],
                    'expected_return': np.sum(mean_returns * result.x),
                    'expected_volatility': np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x))),
                    'expected_sharpe': -result.fun
                })
            except:
                continue
        
        self.rolling_results = pd.DataFrame(rolling_results)
        return self.rolling_results
    
    def plot_analysis(self, figsize=(20, 15)):
        """결과 시각화"""
        if self.returns is None:
            raise ValueError("먼저 분석을 실행해주세요.")
        
        fig = plt.figure(figsize=figsize)
        
        # 1. 가격 차트
        ax1 = plt.subplot(3, 3, 1)
        normalized_prices = self.data / self.data.iloc[0] * 100
        plt.plot(normalized_prices.index, normalized_prices.iloc[:, 0], 
                label=self.symbol1, linewidth=2)
        plt.plot(normalized_prices.index, normalized_prices.iloc[:, 1], 
                label=self.symbol2, linewidth=2)
        plt.title('정규화된 가격 추이 (기준점=100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 효율적 투자선
        if hasattr(self, 'efficient_frontier_data'):
            ax2 = plt.subplot(3, 3, 2)
            ef_data = self.efficient_frontier_data
            plt.scatter(ef_data['annual_volatility'], ef_data['annual_return'], 
                       c=ef_data['sharpe_ratio'], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('연간 변동성 (%)')
            plt.ylabel('연간 수익률 (%)')
            plt.title('효율적 투자선')
            plt.grid(True, alpha=0.3)
        
        # 3. 몬테카를로 결과
        if hasattr(self, 'mc_results'):
            ax3 = plt.subplot(3, 3, 3)
            mc_data = self.mc_results
            plt.scatter(mc_data['annual_volatility'], mc_data['annual_return'], 
                       c=mc_data['sharpe_ratio'], cmap='viridis', alpha=0.3, s=1)
            plt.colorbar(label='Sharpe Ratio')
            plt.xlabel('연간 변동성 (%)')
            plt.ylabel('연간 수익률 (%)')
            plt.title('몬테카를로 시뮬레이션')
            plt.grid(True, alpha=0.3)
        
        # 4. 비중별 샤프 비율
        if hasattr(self, 'efficient_frontier_data'):
            ax4 = plt.subplot(3, 3, 4)
            ef_data = self.efficient_frontier_data
            plt.plot(ef_data['weight1'], ef_data['sharpe_ratio'], linewidth=2)
            plt.xlabel(f'{self.symbol1} 비중')
            plt.ylabel('샤프 비율')
            plt.title('비중별 샤프 비율')
            plt.grid(True, alpha=0.3)
        
        # 5. 비중별 최대 낙폭
        if hasattr(self, 'efficient_frontier_data'):
            ax5 = plt.subplot(3, 3, 5)
            ef_data = self.efficient_frontier_data
            plt.plot(ef_data['weight1'], ef_data['max_drawdown'], 
                    linewidth=2, color='red')
            plt.xlabel(f'{self.symbol1} 비중')
            plt.ylabel('최대 낙폭 (%)')
            plt.title('비중별 최대 낙폭')
            plt.grid(True, alpha=0.3)
        
        # 6. 롤링 최적 비중
        if hasattr(self, 'rolling_results'):
            ax6 = plt.subplot(3, 3, 6)
            rolling_data = self.rolling_results
            plt.plot(rolling_data['date'], rolling_data['weight1'], 
                    linewidth=2, label=f'{self.symbol1} 비중')
            plt.plot(rolling_data['date'], rolling_data['weight2'], 
                    linewidth=2, label=f'{self.symbol2} 비중')
            plt.xlabel('날짜')
            plt.ylabel('비중')
            plt.title('시간별 최적 비중 변화')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # 7. 수익률 분포
        ax7 = plt.subplot(3, 3, 7)
        plt.hist(self.returns.iloc[:, 0], bins=50, alpha=0.7, 
                label=f'{self.symbol1}', density=True)
        plt.hist(self.returns.iloc[:, 1], bins=50, alpha=0.7, 
                label=f'{self.symbol2}', density=True)
        plt.xlabel('일일 수익률')
        plt.ylabel('밀도')
        plt.title('수익률 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 상관관계 히트맵
        ax8 = plt.subplot(3, 3, 8)
        corr_matrix = self.returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[self.symbol1, self.symbol2],
                   yticklabels=[self.symbol1, self.symbol2])
        plt.title('상관관계 히트맵')
        
        # 9. 누적 수익률 비교 (최적 포트폴리오들)
        if hasattr(self, 'results'):
            ax9 = plt.subplot(3, 3, 9)
            
            for name, portfolio in self.results.items():
                cumulative = (1 + portfolio['metrics']['portfolio_returns']).cumprod()
                plt.plot(cumulative.index, cumulative.values, 
                        label=f"{name} ({portfolio['weights'][0]:.1%}/{portfolio['weights'][1]:.1%})",
                        linewidth=2)
            
            plt.xlabel('날짜')
            plt.ylabel('누적 수익률')
            plt.title('최적 포트폴리오 성과 비교')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """결과 요약 출력"""
        print("="*80)
        print(f"📊 {self.symbol1} vs {self.symbol2} 포트폴리오 최적화 결과")
        print("="*80)
        
        if hasattr(self, 'results'):
            print("\n🎯 최적 포트폴리오 비중:")
            print("-"*60)
            
            for name, portfolio in self.results.items():
                w1, w2 = portfolio['weights']
                metrics = portfolio['metrics']
                
                print(f"\n📈 {name.upper().replace('_', ' ')}:")
                print(f"   • {self.symbol1}: {w1:.1%} | {self.symbol2}: {w2:.1%}")
                print(f"   • 연간 수익률: {metrics['annual_return']:.2f}%")
                print(f"   • 연간 변동성: {metrics['annual_volatility']:.2f}%")
                print(f"   • 샤프 비율: {metrics['sharpe_ratio']:.3f}")
                print(f"   • 최대 낙폭: {metrics['max_drawdown']:.2f}%")
        
        if hasattr(self, 'mc_optimal'):
            print(f"\n🎲 몬테카를로 시뮬레이션 최적해:")
            print("-"*60)
            
            best_sharpe = self.mc_optimal['best_sharpe']
            print(f"   • 최고 샤프 비율: {self.symbol1} {best_sharpe['weight1']:.1%} | "
                  f"{self.symbol2} {best_sharpe['weight2']:.1%}")
            print(f"     샤프 비율: {best_sharpe['sharpe_ratio']:.3f}")
        
        print(f"\n📈 개별 자산 통계:")
        print("-"*60)
        for i, symbol in enumerate([self.symbol1, self.symbol2]):
            returns = self.returns.iloc[:, i]
            print(f"   • {symbol}:")
            print(f"     연간 수익률: {returns.mean() * 252 * 100:.2f}%")
            print(f"     연간 변동성: {returns.std() * np.sqrt(252) * 100:.2f}%")
            print(f"     샤프 비율: {(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.3f}")
        
        correlation = self.returns.iloc[:, 0].corr(self.returns.iloc[:, 1])
        print(f"\n🔗 상관계수: {correlation:.3f}")
        
    def run_full_analysis(self, start_date=None, end_date=None, period=None,
                         num_portfolios=100, num_simulations=10000):
        """전체 분석 실행"""
        print("🚀 포트폴리오 최적화 분석을 시작합니다...")
        
        # 1. 데이터 로드 및 필터링
        self.fetch_data()
        
        if self.data is None or self.returns is None:
            print("데이터 로드에 실패하여 분석을 중단합니다.")
            return

        # 기간 필터링
        if start_date and end_date:
            self.data = self.data.loc[start_date:end_date]
        elif period:
            # '1y', '6mo' 등 문자열을 일수로 변환
            def period_to_days(period_str):
                period_str = period_str.lower()
                if period_str.endswith('y'):
                    return int(period_str[:-1]) * 365
                elif period_str.endswith('mo'):
                    return int(period_str[:-2]) * 30
                elif period_str.endswith('w'):
                    return int(period_str[:-1]) * 7
                elif period_str.endswith('d'):
                    return int(period_str[:-1])
                else:
                    raise ValueError(f"지원하지 않는 period 형식: {period_str}")
            days = period_to_days(period)
            from_date = pd.to_datetime('today') - pd.Timedelta(days=days)
            self.data = self.data[self.data.index >= from_date]
        
        # 필터링된 데이터로 수익률 다시 계산
        self.returns = self.data.pct_change().dropna()
        
        print(f"분석 기간: {self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"데이터포인트: {len(self.data)}개")

        # 2. 효율적 투자선 계산
        self.efficient_frontier(num_portfolios)
        
        # 3. 최적 포트폴리오 찾기
        self.find_optimal_portfolios()
        
        # 4. 몬테카를로 시뮬레이션
        self.monte_carlo_optimization(num_simulations)
        
        # 5. 롤링 윈도우 분석
        self.rolling_window_analysis()
        
        # 6. 결과 출력
        self.print_summary()
        
        # 7. 시각화
        self.plot_analysis()
        
        return self.results

# 사용 예시
if __name__ == "__main__":
    # ETH-DOT 포트폴리오 최적화
    optimizer = PortfolioOptimizer('ETH', 'DOT')
    
    # 전체 분석 실행
    results = optimizer.run_full_analysis(
        period='1y',           # 최근 1년
        num_portfolios=100,    # 효율적 투자선 포인트 수
        num_simulations=10000  # 몬테카를로 시뮬레이션 횟수
    )
    
    # 특정 기간 분석
    # print("\n" + "="*80)
    # print("📅 특정 기간 분석 예시")
    
    # optimizer2 = PortfolioOptimizer('ETH', 'DOT')
    # results2 = optimizer2.run_full_analysis(
    #     start_date='2023-01-01',
    #     end_date='2024-01-01'
    # )
    
    # 다른 자산 조합 분석
    # print("\n" + "="*80)
    # print("🔄 다른 자산 조합 분석")
    
    # optimizer3 = PortfolioOptimizer('BTC', 'ETH')
    # results3 = optimizer3.run_full_analysis(period='6mo')