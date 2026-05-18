import run_all_validation as r
root=r.root_dir(); oso,p=r.run_oso_loro(root); (root/'outputs').mkdir(exist_ok=True); oso.to_csv(root/'outputs'/'oso_loro_metrics.csv',index=False); p.to_csv(root/'outputs'/'oso_loro_predictions.csv',index=False); print('wrote Oso LORO outputs')
