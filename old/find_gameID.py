from nba_api.stats.endpoints import leaguegamefinder

# LeagueGameFinderを使って特定のシーズンの試合情報を取得
game_finder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24')

# データを取得し、pandas DataFrameに変換
games_dict = game_finder.get_data_frames()[0]

# 直近5試合の情報を表示してみる
print(games_dict.head(5)[['GAME_ID', 'GAME_DATE', 'MATCHUP']])