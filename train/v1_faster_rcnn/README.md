### 批量整理评估输出文件
cd eval_result_all_epoch
mkdir -p confusion_matrix metrics
mv *.jpg confusion_matrix/
mv *.txt metrics/