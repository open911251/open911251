import pandas as pd

# 檔案路徑
file_path = 'C:\\Users\\Peter\\Desktop\\csv\\ddt\\'

# 載入數據
ddt_log = pd.read_csv(file_path + '5_ddt_log.csv')
ddt_blog_comment = pd.read_csv(file_path + '8_ddt_blog_comment.csv')
ddt_content = pd.read_csv(file_path + '0_ddt_content.csv')

# 步驟 1: 標記新成員
# 將 CREATION_DATE 轉換為 datetime 格式
ddt_log['CREATION_DATE'] = pd.to_datetime(ddt_log['CREATION_DATE'], format='%Y/%m/%d %H:%M')

# 找到每個 USER_ID 的首次行為時間
first_activity = ddt_log.groupby('USER_ID')['CREATION_DATE'].min().reset_index()
first_activity.columns = ['USER_ID', 'FIRST_ACTIVITY_DATE']

# 定義 "新成員" 的門檻 (如最近 30 天內的首次活動)
new_member_threshold = first_activity['FIRST_ACTIVITY_DATE'].max() - pd.Timedelta(days=30)
new_members = first_activity[first_activity['FIRST_ACTIVITY_DATE'] >= new_member_threshold]

# 步驟 2: 篩選新成員的瀏覽和互動行為
new_member_ids = new_members['USER_ID'].unique()
new_member_logs = ddt_log[ddt_log['USER_ID'].isin(new_member_ids)]

# 步驟 3: 分析新成員與評論的互動
# 假設 ACTION_NAME 中包含 'View' 表示瀏覽行為
new_member_comments = new_member_logs[new_member_logs['ACTION_NAME'].str.contains('View', na=False)]

# 計算每個評論被新成員瀏覽或互動的次數
comment_interaction_count = new_member_comments['OBJECT_ID'].value_counts().reset_index()
comment_interaction_count.columns = ['BLOG_ID', 'INTERACTION_COUNT']

# 將結果與評論數據合併，根據 BLOG_ID 匹配
top_comments = comment_interaction_count.merge(ddt_blog_comment, left_on='BLOG_ID', right_on='BLOG_ID')

# 步驟 4: 分析新成員最常互動的內容
# 計算每篇內容被新成員瀏覽的次數
content_interaction_count = new_member_logs['OBJECT_ID'].value_counts().reset_index()
content_interaction_count.columns = ['CONTENT_ID', 'VIEW_COUNT']

# 將結果與內容數據合併
top_contents = content_interaction_count.merge(ddt_content, left_on='CONTENT_ID', right_on='CONTENT_ID')

# 顯示結果
print("新成員數量:", new_members.shape[0])
print("最具影響力的評論前5名:")
print(top_comments.head())
print("最受歡迎的內容前5名:")
print(top_contents.head())
