# 将该数据导出为excel，可以使用pandas
import pandas as pd

# 读取txt文件，自动以制表符分隔
df = pd.read_csv('/data/sfs/home/rensiyu/Attention/Abstract-Attention-Map/eval/eval_results2/screenspot-Pro_all_preds_StandardResize.txt', sep='\t')

# 导出为excel
df.to_excel('/data/sfs/home/rensiyu/Attention/Abstract-Attention-Map/eval/eval_results2/screenspot-Pro_all_preds_StandardResize.xlsx', index=False)