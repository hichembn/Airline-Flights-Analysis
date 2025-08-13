import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AirlineMarketAnalysis:
    def __init__(self, data_path):
        """Initialize the analysis with airline dataset"""
        self.data = pd.read_csv(data_path)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the dataset"""
        print("🧹 Cleaning dataset...")
        
        # Remove any unnamed columns
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
        
        # Strip whitespace from column names
        self.data.columns = self.data.columns.str.strip()
        
        # Convert price to numeric, handling any string values
        self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')
        
        # Convert days_left to numeric
        self.data['days_left'] = pd.to_numeric(self.data['days_left'], errors='coerce')
        
        # Create route column for easier analysis
        self.data['route'] = self.data['source_city'].astype(str) + ' → ' + self.data['destination_city'].astype(str)
        
        # Remove rows with missing essential data
        essential_cols = ['airline', 'price', 'route']
        self.data = self.data.dropna(subset=essential_cols)
        
        print(f"✅ Data cleaned. Shape: {self.data.shape}")
        
    def basic_overview(self):
        """Generate comprehensive dataset overview"""
        print("\n" + "="*60)
        print("📊 AIRLINE MARKET DATASET OVERVIEW")
        print("="*60)
        
        # Basic statistics
        print(f"Dataset Size: {self.data.shape[0]:,} flights, {self.data.shape[1]} variables")
        print(f"Date Range: {self.data['days_left'].min()}-{self.data['days_left'].max()} days ahead")
        print(f"Price Range: ${self.data['price'].min():.0f} - ${self.data['price'].max():,.0f}")
        
        # Market structure
        print(f"\n🏢 Market Structure:")
        print(f"   • {self.data['airline'].nunique()} airlines")
        print(f"   • {self.data['route'].nunique()} unique routes") 
        print(f"   • {self.data['class'].nunique()} service classes")
        print(f"   • {self.data['source_city'].nunique()} origin cities")
        print(f"   • {self.data['destination_city'].nunique()} destination cities")
        
        # Missing data analysis
        print(f"\n🔍 Data Quality:")
        missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
        for col in missing_pct[missing_pct > 0].index:
            print(f"   • {col}: {missing_pct[col]:.1f}% missing")
            
        return self.data.describe()
    
    def market_concentration_analysis(self):
        """Analyze market concentration and competitive dynamics"""
        print("\n" + "="*60)
        print("🏆 MARKET CONCENTRATION ANALYSIS")
        print("="*60)
        
        # Overall airline market share
        airline_share = (self.data['airline'].value_counts() / len(self.data) * 100).round(2)
        print("\n📈 Overall Market Share:")
        for airline, share in airline_share.head(10).items():
            print(f"   {airline}: {share}%")
            
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(airline_share ** 2)
        print(f"\n📊 Market Concentration (HHI): {hhi:.0f}")
        if hhi > 2500:
            print("   Status: Highly concentrated market")
        elif hhi > 1500:
            print("   Status: Moderately concentrated market") 
        else:
            print("   Status: Competitive market")
            
        # Route-level concentration
        route_concentration = {}
        for route in self.data['route'].value_counts().head(20).index:
            route_data = self.data[self.data['route'] == route]
            route_airlines = route_data['airline'].value_counts()
            route_hhi = sum((route_airlines / len(route_data) * 100) ** 2)
            route_concentration[route] = {
                'flights': len(route_data),
                'airlines': len(route_airlines),
                'hhi': route_hhi,
                'dominant_airline': route_airlines.index[0],
                'dominant_share': route_airlines.iloc[0] / len(route_data) * 100
            }
            
        # Display most concentrated routes
        print(f"\n🛣️ Most Concentrated Routes (Top 10):")
        sorted_routes = sorted(route_concentration.items(), 
                             key=lambda x: x[1]['hhi'], reverse=True)[:10]
        
        for route, metrics in sorted_routes:
            print(f"   {route}")
            print(f"     Dominant: {metrics['dominant_airline']} ({metrics['dominant_share']:.1f}%)")
            print(f"     Airlines: {metrics['airlines']}, HHI: {metrics['hhi']:.0f}")
            
        return airline_share, route_concentration
    
    def pricing_strategy_analysis(self):
        """Analyze pricing strategies across segments"""
        print("\n" + "="*60)
        print("💰 PRICING STRATEGY ANALYSIS")
        print("="*60)
        
        # Price by airline
        airline_pricing = self.data.groupby('airline')['price'].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        airline_pricing.columns = ['Flights', 'Avg_Price', 'Median_Price', 'Price_StdDev']
        airline_pricing = airline_pricing.sort_values('Avg_Price', ascending=False)
        
        print("\n💸 Airline Pricing Tiers (Top 10):")
        for airline, row in airline_pricing.head(10).iterrows():
            tier = "Premium" if row['Avg_Price'] > airline_pricing['Avg_Price'].quantile(0.75) else \
                   "Budget" if row['Avg_Price'] < airline_pricing['Avg_Price'].quantile(0.25) else "Mid-tier"
            print(f"   {airline}: ${row['Avg_Price']:.0f} ({tier})")
            
        # Price by class
        class_pricing = self.data.groupby('class')['price'].agg(['mean', 'count']).round(2)
        class_pricing.columns = ['Avg_Price', 'Flights']
        print(f"\n✈️ Price by Service Class:")
        for class_name, row in class_pricing.sort_values('Avg_Price', ascending=False).iterrows():
            print(f"   {class_name}: ${row['Avg_Price']:.0f} ({row['Flights']} flights)")
            
        # Price by stops
        stops_pricing = self.data.groupby('stops')['price'].agg(['mean', 'count']).round(2)
        print(f"\n🔄 Price by Number of Stops:")
        for stops, row in stops_pricing.iterrows():
            print(f"   {stops} stops: ${row['mean']:.0f} ({row['count']} flights)")
            
        return airline_pricing, class_pricing, stops_pricing
    
    def route_opportunity_analysis(self):
        """Identify route opportunities and market gaps"""
        print("\n" + "="*60)
        print("🎯 ROUTE OPPORTUNITY ANALYSIS")
        print("="*60)
        
        # High-value routes (price per mile proxy)
        route_metrics = self.data.groupby('route').agg({
            'price': ['mean', 'count', 'std'],
            'airline': 'nunique',
            'stops': 'mean'
        }).round(2)
        
        route_metrics.columns = ['Avg_Price', 'Flight_Count', 'Price_StdDev', 'Num_Airlines', 'Avg_Stops']
        route_metrics['Price_Premium'] = (route_metrics['Avg_Price'] - self.data['price'].mean()) / self.data['price'].std()
        
        # High-value, low-competition routes (opportunities)
        opportunities = route_metrics[
            (route_metrics['Price_Premium'] > 1) & 
            (route_metrics['Num_Airlines'] <= 2) &
            (route_metrics['Flight_Count'] >= 10)
        ].sort_values('Price_Premium', ascending=False)
        
        print(f"\n🚀 Market Opportunities (High price, low competition):")
        for route, metrics in opportunities.head(10).iterrows():
            print(f"   {route}")
            print(f"     Price: ${metrics['Avg_Price']:.0f} (+{metrics['Price_Premium']:.1f}σ)")
            print(f"     Airlines: {metrics['Num_Airlines']}, Flights: {metrics['Flight_Count']}")
            
        # Underserved routes (high demand, many stops)
        underserved = route_metrics[
            (route_metrics['Flight_Count'] >= 20) & 
            (route_metrics['Avg_Stops'] >= 1)
        ].sort_values(['Flight_Count', 'Avg_Stops'], ascending=[False, False])
        
        print(f"\n🔄 Potentially Underserved Routes (Many connections needed):")
        for route, metrics in underserved.head(10).iterrows():
            print(f"   {route}")
            print(f"     Demand: {metrics['Flight_Count']} flights, Avg stops: {metrics['Avg_Stops']:.1f}")
            print(f"     Airlines serving: {metrics['Num_Airlines']}")
            
        return opportunities, underserved
    
    def customer_segmentation_analysis(self):
        """Analyze customer segments and behavior patterns"""
        print("\n" + "="*60)
        print("👥 CUSTOMER SEGMENTATION ANALYSIS")
        print("="*60)
        
        # Booking behavior by advance purchase
        self.data['booking_window'] = pd.cut(self.data['days_left'], 
                                           bins=[0, 7, 30, 90, float('inf')],
                                           labels=['Last_minute', 'Short_term', 'Medium_term', 'Long_term'])
        
        booking_behavior = self.data.groupby('booking_window').agg({
            'price': ['mean', 'count'],
            'class': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'stops': 'mean'
        }).round(2)
        
        print(f"\n📅 Booking Behavior Analysis:")
        for window, metrics in booking_behavior.iterrows():
            avg_price = metrics[('price', 'mean')]
            count = metrics[('price', 'count')]
            popular_class = metrics[('class', '<lambda>')]
            avg_stops = metrics[('stops', 'mean')]
            print(f"   {window} ({count} bookings):")
            print(f"     Avg price: ${avg_price:.0f}, Popular class: {popular_class}")
            print(f"     Avg stops: {avg_stops:.1f}")
            
        # Price sensitivity by segment
        segment_analysis = self.data.groupby(['class', 'booking_window'])['price'].agg(['mean', 'count']).round(2)
        
        return booking_behavior, segment_analysis
    
    def create_visualizations(self):
        """Create comprehensive visualization suite"""
        print("\n" + "="*60)
        print("📊 GENERATING MARKET VISUALIZATIONS")
        print("="*60)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Market Share
        plt.subplot(2, 3, 1)
        airline_counts = self.data['airline'].value_counts().head(10)
        plt.pie(airline_counts.values, labels=airline_counts.index, autopct='%1.1f%%')
        plt.title('Market Share by Airline', fontsize=14, fontweight='bold')
        
        # 2. Price Distribution by Class
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.data, x='class', y='price')
        plt.title('Price Distribution by Class', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 3. Price vs Days Left
        plt.subplot(2, 3, 3)
        plt.scatter(self.data['days_left'], self.data['price'], alpha=0.5)
        plt.xlabel('Days Until Departure')
        plt.ylabel('Price ($)')
        plt.title('Price vs Booking Window', fontsize=14, fontweight='bold')
        
        # 4. Average Price by Number of Stops
        plt.subplot(2, 3, 4)
        stops_avg = self.data.groupby('stops')['price'].mean()
        stops_avg.plot(kind='bar')
        plt.title('Average Price by Stops', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Stops')
        plt.ylabel('Average Price ($)')
        
        # 5. Route Volume Distribution
        plt.subplot(2, 3, 5)
        route_counts = self.data['route'].value_counts().head(15)
        plt.barh(range(len(route_counts)), route_counts.values)
        plt.yticks(range(len(route_counts)), route_counts.index)
        plt.title('Top 15 Routes by Volume', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Flights')
        
        # 6. Price Heatmap by Airline and Class
        plt.subplot(2, 3, 6)
        pivot_table = self.data.pivot_table(values='price', index='airline', columns='class', aggfunc='mean')
        sns.heatmap(pivot_table.head(10), annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Price Heatmap: Airlines vs Classes', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('airline_market_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'airline_market_analysis.png'")
    
    def generate_strategic_insights(self):
        """Generate actionable business insights"""
        print("\n" + "="*60)
        print("🎯 STRATEGIC MARKET INSIGHTS")
        print("="*60)
        
        insights = []
        
        # Market concentration insight
        airline_share = self.data['airline'].value_counts(normalize=True) * 100
        top3_share = airline_share.head(3).sum()
        insights.append(f"Market Concentration: Top 3 airlines control {top3_share:.1f}% of flights")
        
        # Price premium opportunities
        class_premiums = self.data.groupby('class')['price'].mean().sort_values(ascending=False)
        if len(class_premiums) > 1:
            premium_opportunity = ((class_premiums.iloc[0] - class_premiums.iloc[-1]) / class_premiums.iloc[-1]) * 100
            insights.append(f"Premium Pricing: {class_premiums.index[0]} commands {premium_opportunity:.1f}% premium over {class_premiums.index[-1]}")
        
        # Route efficiency insight
        direct_vs_stops = self.data.groupby('stops')['price'].mean()
        if len(direct_vs_stops) > 1:
            connection_penalty = ((direct_vs_stops.iloc[-1] - direct_vs_stops.iloc[0]) / direct_vs_stops.iloc[0]) * 100
            insights.append(f"Connection Premium: Flights with stops cost {abs(connection_penalty):.1f}% {'more' if connection_penalty > 0 else 'less'}")
        
        # Booking window insight
        booking_corr = self.data['days_left'].corr(self.data['price'])
        trend = "increase" if booking_corr > 0 else "decrease"
        insights.append(f"Booking Timing: Prices tend to {trend} as departure approaches (correlation: {booking_corr:.2f})")
        
        print("\n💡 Key Strategic Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
            
        return insights
    
    def run_full_analysis(self):
        """Execute complete market analysis"""
        print("🚀 Starting Comprehensive Airline Market Analysis...")
        
        # Run all analyses
        overview = self.basic_overview()
        airline_share, route_conc = self.market_concentration_analysis()
        pricing_analysis = self.pricing_strategy_analysis()
        opportunities, underserved = self.route_opportunity_analysis()
        segmentation = self.customer_segmentation_analysis()
        
        # Generate visualizations
        self.create_visualizations()
        
        # Strategic insights
        insights = self.generate_strategic_insights()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("📁 Files generated:")
        print("   • airline_market_analysis.png (comprehensive charts)")
        print("\n🎯 Next Steps for Portfolio:")
        print("   1. Deep dive into identified opportunities")
        print("   2. Develop competitive positioning strategies") 
        print("   3. Create route optimization recommendations")
        print("   4. Build predictive pricing models")
        
        return {
            'overview': overview,
            'market_concentration': (airline_share, route_conc),
            'pricing_analysis': pricing_analysis,
            'opportunities': opportunities,
            'underserved_routes': underserved,
            'segmentation': segmentation,
            'insights': insights
        }

# Usage Example
if __name__ == "__main__":
    # Initialize analysis
    # analyzer = AirlineMarketAnalysis('your_airline_dataset.csv')
    
    # Run complete analysis
    # results = analyzer.run_full_analysis()
    
    print("📝 To use this analysis:")
    print("1. Replace 'your_airline_dataset.csv' with your file path")
    print("2. Run: python airline_analysis.py")
    print("3. Review generated insights and visualizations")
    print("\n🎯 This analysis will provide:")
    print("   • Market concentration metrics")
    print("   • Competitive landscape mapping")
    print("   • Route opportunity identification")
    print("   • Customer segmentation insights")
    print("   • Strategic recommendations")