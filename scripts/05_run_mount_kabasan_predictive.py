import run_all_validation as r
root=r.root_dir(); m,p=r.run_mount(root); (root/'outputs').mkdir(exist_ok=True); m.to_csv(root/'outputs'/'mount_kabasan_predictive_metrics.csv',index=False); p.to_csv(root/'outputs'/'mount_kabasan_predictive_predictions.csv',index=False); print('wrote Mount Kaba-san outputs')
