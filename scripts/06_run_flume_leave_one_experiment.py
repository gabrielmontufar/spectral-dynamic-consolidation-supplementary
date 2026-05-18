import run_all_validation as r
root=r.root_dir(); m,p=r.run_flume(root); (root/'outputs').mkdir(exist_ok=True); m.to_csv(root/'outputs'/'flume_leave_one_experiment_metrics.csv',index=False); p.to_csv(root/'outputs'/'flume_leave_one_experiment_predictions.csv',index=False); print('wrote flume outputs')
