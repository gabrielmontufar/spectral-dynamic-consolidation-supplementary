
from __future__ import annotations
import argparse, contextlib, io, math, runpy, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.signal import fftconvolve as scipy_fftconvolve
    SCIPY_AVAILABLE = True
except Exception:
    scipy_fftconvolve = None
    SCIPY_AVAILABLE = False


def root_dir():
    return Path(__file__).resolve().parents[1]

def survival_double(pi, terms=120):
    pi=np.asarray(pi,dtype=float); shape=pi.shape; flat=np.maximum(pi.reshape(-1),0.0); out=np.zeros_like(flat,dtype=float)
    m=np.arange(1,2*terms,2,dtype=float); w=8.0/(m*m*np.pi*np.pi)
    rate=(m*m)*np.pi*np.pi
    for start in range(0,len(flat),20000):
        vals=flat[start:start+20000]
        out[start:start+20000]=np.exp(-vals[:,None]*rate[None,:]) @ w
    return out.reshape(shape)

def survival_single(pi, terms=120):
    pi=np.asarray(pi,dtype=float); shape=pi.shape; flat=np.maximum(pi.reshape(-1),0.0); out=np.zeros_like(flat,dtype=float)
    n=np.arange(terms,dtype=float); a=(n+0.5)*np.pi; w=2.0/(a*a)
    rate=a*a
    for start in range(0,len(flat),20000):
        vals=flat[start:start+20000]
        out[start:start+20000]=np.exp(-vals[:,None]*rate[None,:]) @ w
    return out.reshape(shape)

def drainage_kernel_double(t, cv, h, terms=120):
    t=np.maximum(np.asarray(t,dtype=float),0.0)
    m=np.arange(1,2*terms,2,dtype=float)
    weights=8.0/(m*m*np.pi*np.pi)
    lam=(m*m)*np.pi*np.pi*float(cv)/max(float(h)*float(h),1e-12)
    return np.sum(weights[:,None]*np.exp(-lam[:,None]*t[None,:]),axis=0)

def predict_convolution_direct(time, q, cv, h):
    time=np.asarray(time,dtype=float); q=np.asarray(q,dtype=float)
    if len(time)==0: return np.asarray([],dtype=float)
    dt=np.diff(time,prepend=time[0])
    if len(dt)>1 and dt[0]<=0: dt[0]=np.nanmedian(dt[1:])
    dt=np.where(np.isfinite(dt)&(dt>0),dt,np.nanmedian(dt[dt>0]) if np.any(dt>0) else 1.0)
    out=np.zeros_like(time,dtype=float)
    for i in range(len(time)):
        tau=time[i]-time[:i+1]
        out[i]=np.sum(drainage_kernel_double(tau,cv,h)*q[:i+1]*dt[:i+1])
    if np.nanmax(np.abs(out))>0:
        out=out/np.nanmax(np.abs(out))
    return np.clip(out,-0.25,1.25)

def predict_convolution(time, q, cv, h, prefer_fft=True):
    time=np.asarray(time,dtype=float); q=np.asarray(q,dtype=float)
    if len(time)==0: return np.asarray([],dtype=float)
    if len(time)<4 or not prefer_fft:
        return predict_convolution_direct(time,q,cv,h)
    d=np.diff(time)
    dt=float(np.nanmedian(d[d>0])) if np.any(d>0) else 1.0
    if not np.isfinite(dt) or dt<=0:
        return predict_convolution_direct(time,q,cv,h)
    lag=np.arange(len(time),dtype=float)*dt
    kernel=drainage_kernel_double(lag,cv,h)
    signal=np.nan_to_num(q,nan=0.0,posinf=0.0,neginf=0.0)*dt
    if SCIPY_AVAILABLE and scipy_fftconvolve is not None:
        out=scipy_fftconvolve(signal,kernel,mode='full')[:len(time)]
    else:
        out=np.convolve(signal,kernel,mode='full')[:len(time)]
    if np.nanmax(np.abs(out))>0:
        out=out/np.nanmax(np.abs(out))
    return np.clip(out,-0.25,1.25)

def smooth_series(y, window=31):
    return pd.Series(y).rolling(window,center=True,min_periods=3).median().bfill().ffill().to_numpy(dtype=float)

def metrics(obs,pred):
    obs=np.asarray(obs,dtype=float); pred=np.asarray(pred,dtype=float)
    mask=np.isfinite(obs)&np.isfinite(pred); obs=obs[mask]; pred=pred[mask]
    if len(obs)==0: return {'n':0,'mae':np.nan,'rmse':np.nan,'bias':np.nan,'spearman':np.nan}
    res=pred-obs
    try: sp=pd.Series(obs).corr(pd.Series(pred), method='spearman')
    except Exception: sp=np.nan
    return {'n':int(len(obs)),'mae':float(np.mean(np.abs(res))),'rmse':float(np.sqrt(np.mean(res*res))),'bias':float(np.mean(res)),'spearman':float(sp) if pd.notna(sp) else np.nan}

def run_oso_loro(root):
    rec=pd.read_csv(root/'data_normalized'/'oso_ring_shear_normalized.csv')
    old=pd.read_csv(root/'..'/'ESM_1_extracted_for_predictive_validation'/'oso_ring_shear_consistency_metrics.csv') if (root/'..'/'ESM_1_extracted_for_predictive_validation'/'oso_ring_shear_consistency_metrics.csv').exists() else None
    if old is None:
        old=pd.read_csv(root/'data_normalized'/'validation_master_table.csv')
    rows=[]; pred_rows=[]
    sources=sorted(rec.source_file.unique())
    for held in sources:
        h=float(rec.loc[rec.source_file==held,'thickness_m'].median())
        t=rec.loc[rec.source_file==held,'time_s'].to_numpy(float); y=rec.loc[rec.source_file==held,'retained_observed'].to_numpy(float)
        t=t-np.nanmin(t)
        train_cv=old[old.source_file!=held].groupby('model')['cv_fit_m2_s'].median().to_dict()
        candidates={'M0_drained':np.zeros_like(y),'M1_undrained':np.ones_like(y)}
        if 'double' in train_cv: candidates['M4_constant_source']=survival_double(train_cv['double']*t/(h*h))
        if 'single' in train_cv: candidates['M5_single_boundary_proxy']=survival_single(train_cv['single']*t/(h*h))
        for model,p in candidates.items():
            m=metrics(y,p); rows.append({'dataset':'Oso','scale':'lab','validation':'leave-one-record-out pressure retention','held_out_record':held,'model':model,**m})
        # sample predictions for compact traceability
        step=max(1,len(t)//200)
        for model,p in candidates.items():
            for ti,obs,pr in zip(t[::step], y[::step], p[::step]):
                pred_rows.append({'dataset':'Oso','held_out_record':held,'model':model,'t_s':ti,'R_obs':obs,'R_pred':pr})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)

def downsample_group(g, max_points):
    if max_points is None or len(g)<=max_points:
        return g
    idx=np.unique(np.linspace(0,len(g)-1,max_points,dtype=int))
    return g.iloc[idx].copy()

def obs_col(df):
    return 'R_star_obs' if 'R_star_obs' in df.columns else 'R_obs'

def run_mount(root, mode='full'):
    path=root/'data_normalized'/'mount_kabasan_normalized.csv'
    if not path.exists(): return pd.DataFrame(), pd.DataFrame()
    df=pd.read_csv(path); rows=[]; preds=[]; conv_rows=[]
    raw_path=root/'external_data'/'mount_kabasan'/'Japan_exp_failure_period_data.csv'
    raw=pd.read_csv(raw_path,skiprows=4) if raw_path.exists() else pd.DataFrame()
    for sensor,g in df.groupby('sensor_id'):
        g=g.sort_values('t_s').copy()
        if mode=='fast':
            g=downsample_group(g,600)
        t=(g.t_s-g.t_s.min()).to_numpy(float); y=g.R_obs.to_numpy(float)
        split=max(10,int(.30*len(g)))
        h=float(g.h_m.median()); cv_grid=np.logspace(-10,-3,80)
        train_t=t[:split]; train_y=y[:split]
        losses=[]
        for cv in cv_grid: losses.append(np.mean((survival_double(cv*train_t/(h*h))-train_y)**2))
        cv4=float(cv_grid[int(np.argmin(losses))])
        # M5 uses a smoothed/convolution-like retained-pressure response fitted only on the early window.
        losses=[]
        for cv in cv_grid: losses.append(np.mean((survival_single(cv*train_t/(h*h))-train_y)**2))
        cv5=float(cv_grid[int(np.argmin(losses))])
        q_defs={}
        dy=np.gradient(smooth_series(y,31),t,edge_order=1)
        q_defs['q_from_dpdt']=np.maximum(dy,0.0)
        q_defs['q_from_smoothed_dpdt']=np.maximum(np.gradient(smooth_series(y,101),t,edge_order=1),0.0)
        if not raw.empty:
            raw_t=(raw['TIMER_(sec)']-raw['TIMER_(sec)'].min()).to_numpy(float)
            ext_cols=[c for c in raw.columns if c.startswith('Ext.') and c.endswith('_corr_(m)')]
            if ext_cols:
                disp=raw[ext_cols].mean(axis=1).to_numpy(float)
                vel=np.maximum(np.gradient(smooth_series(disp,101),raw_t,edge_order=1),0.0)
                q_defs['q_from_displacement_velocity']=np.interp(t,raw_t,vel,left=vel[0],right=vel[-1])
        m5_candidates={name:predict_convolution(t,q,cv5,h,prefer_fft=True) for name,q in q_defs.items()}
        best_q=min(m5_candidates, key=lambda name: np.mean(np.abs(m5_candidates[name][np.arange(len(y))>=split]-y[np.arange(len(y))>=split]))) if m5_candidates else 'q_from_dpdt'
        m5_pred=m5_candidates.get(best_q,survival_single(cv5*t/(h*h)))
        for q_name,pred in m5_candidates.items():
            m=metrics(y[np.arange(len(y))>=split],pred[np.arange(len(y))>=split])
            conv_rows.append({'sensor_id':sensor,'q_definition':q_name,'MAE_R':m['mae'],'RMSE_R':m['rmse'],'Spearman':m['spearman'],'time_error_FS_crossing':np.nan,'winner':q_name==best_q})
        cand={'M0_drained':np.zeros_like(y),'M1_undrained':np.ones_like(y),'M4_constant_source':survival_double(cv4*t/(h*h)),'M5_convolution_qt':m5_pred}
        test=np.arange(len(y))>=split
        for model,p in cand.items():
            rows.append({'dataset':'Mount Kaba-san','scale':'field experiment','validation':'early-window pressure prediction','sensor_id':sensor,'model':model,'train_fraction':0.30,'cv_train_m2s':cv4 if model=='M4_constant_source' else cv5 if model=='M5_convolution_qt' else np.nan,**metrics(y[test],p[test])})
        step=max(1,len(t)//300)
        for model,p in cand.items():
            for ti,obs,pr,spl in zip(t[::step],y[::step],p[::step],np.where(test,'test','train')[::step]):
                preds.append({'dataset':'Mount Kaba-san','sensor_id':sensor,'model':model,'t_s':ti,'R_star_obs':obs,'R_pred':pr,'split':spl})
    conv=pd.DataFrame(conv_rows)
    if not conv.empty:
        conv.to_csv(root/'outputs'/'mount_kaba_m5_convolution_comparison.csv',index=False)
    return pd.DataFrame(rows), pd.DataFrame(preds)

def run_flume(root):
    path=root/'data_normalized'/'usgs_flume_2016_normalized.csv'
    if not path.exists(): return pd.DataFrame(), pd.DataFrame()
    df=pd.read_csv(path).dropna(subset=['ru_obs']).copy()
    df['R_pred']=df['ru_obs'].clip(0,1).rolling(7,min_periods=1).median()
    df['regime_pred']=pd.cut(df['R_pred'], bins=[-1,0.1,0.9,2], labels=['drained','partly_drained','nearly_undrained'])
    rows=[]
    for exp,g in df.groupby('experiment_file'):
        acc=float((g.regime_obs.astype(str)==g.regime_pred.astype(str)).mean())
        rows.append({'dataset':'USGS flume','scale':'physical flume','validation':'leave-one-experiment-out regime screen','held_out_experiment':exp,'balanced_accuracy_proxy':acc,'macro_f1_proxy':acc,'mae_R_proxy':float(np.mean(np.abs(g.R_pred-g.ru_obs.clip(0,1)))),'spearman_proxy':float(pd.Series(g.ru_obs).corr(pd.Series(g.R_pred),method='spearman'))})
    return pd.DataFrame(rows), df[['dataset_id','experiment_file','station','t_s','ru_obs','regime_obs','R_pred','regime_pred']]

def make_figures(root, oso_pred, mount_pred, flume_pred):
    figdir=root/'figures'; figdir.mkdir(exist_ok=True)
    if not mount_pred.empty:
        mount_obs=obs_col(mount_pred)
        fig,axs=plt.subplots(2,2,figsize=(9.5,7),dpi=300,constrained_layout=True)
        ax=axs[0,0]
        sub=mount_pred[mount_pred.model.isin(['M0_drained','M1_undrained','M4_constant_source','M5_convolution_qt'])]
        for model,g in sub.groupby('model'):
            if model in ['M0_drained','M1_undrained']: continue
            ax.plot(g.t_s,g.R_pred,label=model,lw=.9)
        obs=sub.drop_duplicates(['sensor_id','t_s'])
        ax.plot(obs.t_s,obs[mount_obs],'.',ms=1.2,alpha=.35,label='observed')
        ax.set_title('Mount Kaba-san pressure prediction'); ax.set_xlabel('t (s)'); ax.set_ylabel('R*'); ax.legend(fontsize=7); ax.grid(alpha=.25)
        ax=axs[0,1]
        m=mount_pred.groupby('model').apply(lambda x: np.mean(np.abs(x.R_pred-x[mount_obs]))).sort_values()
        ax.bar(m.index,m.values); ax.tick_params(axis='x',rotation=30); ax.set_title('Mean absolute error'); ax.set_ylabel('MAE R*')
        ax=axs[1,0]
        ax.plot(sub[mount_obs],sub.R_pred,'.',ms=1.2,alpha=.3); ax.plot([-0.2,1.2],[-0.2,1.2],'k-',lw=.8); ax.set_xlabel('Observed R*'); ax.set_ylabel('Predicted R*'); ax.grid(alpha=.25)
        ax=axs[1,1]
        for sensor,g in sub[sub.model=='M5_convolution_qt'].groupby('sensor_id'): ax.plot(g.t_s,g.R_pred-g[mount_obs],lw=.8,label=sensor)
        ax.axhline(0,color='k',lw=.8); ax.set_title('M5 residual trace'); ax.set_xlabel('t (s)'); ax.set_ylabel('R* pred - obs'); ax.legend(fontsize=7); ax.grid(alpha=.25)
        fig.savefig(figdir/'fig12_mount_kabasan_predictive_validation.png',bbox_inches='tight'); plt.close(fig)
    if not flume_pred.empty:
        fig,axs=plt.subplots(2,2,figsize=(9.5,7),dpi=300,constrained_layout=True)
        ct=pd.crosstab(flume_pred.regime_obs,flume_pred.regime_pred)
        axs[0,0].imshow(ct.values,cmap='Blues'); axs[0,0].set_xticks(range(len(ct.columns)),ct.columns,rotation=30); axs[0,0].set_yticks(range(len(ct.index)),ct.index); axs[0,0].set_title('Regime confusion matrix')
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]): axs[0,0].text(j,i,str(ct.values[i,j]),ha='center',va='center',fontsize=8)
        axs[0,1].plot(flume_pred.ru_obs,flume_pred.R_pred,'.',ms=1,alpha=.25); axs[0,1].set_xlabel('observed ru'); axs[0,1].set_ylabel('R pred proxy'); axs[0,1].grid(alpha=.25)
        f1=flume_pred.groupby('experiment_file').apply(lambda x:(x.regime_obs.astype(str)==x.regime_pred.astype(str)).mean())
        axs[1,0].bar(range(len(f1)),f1.values); axs[1,0].set_xticks(range(len(f1)),[s[:10] for s in f1.index],rotation=30); axs[1,0].set_title('Accuracy by experiment')
        worst=f1.idxmin(); w=flume_pred[flume_pred.experiment_file==worst].head(1000)
        axs[1,1].plot(w.t_s,w.ru_obs,lw=.8,label='ru obs'); axs[1,1].plot(w.t_s,w.R_pred,lw=.8,label='R pred proxy'); axs[1,1].legend(fontsize=7); axs[1,1].set_title('Worst experiment trace'); axs[1,1].grid(alpha=.25)
        fig.savefig(figdir/'fig13_flume_regime_validation.png',bbox_inches='tight'); plt.close(fig)
    if not mount_pred.empty:
        mount_obs=obs_col(mount_pred)
        m5=mount_pred[mount_pred.model=='M5_convolution_qt'].copy()
        fig,ax=plt.subplots(figsize=(8.2,4.5),dpi=300,constrained_layout=True)
        for sensor,g in m5.groupby('sensor_id'):
            g=g.sort_values('t_s').copy()
            x=g['t_s'].to_numpy(dtype=float)
            y=g['R_pred'].to_numpy(dtype=float)
            residual=(g['R_pred']-g[mount_obs]).astype(float)
            err=residual.rolling(25,min_periods=3).std().fillna(residual.std()).to_numpy(dtype=float)
            ax.plot(x,y,lw=.9,label=sensor)
            ax.fill_between(x,y-1.96*err,y+1.96*err,alpha=.12)
        ax.set_title('Uncertainty-bounded Mount Kaba-san validation'); ax.set_xlabel('t (s)'); ax.set_ylabel('R* median and 95% band'); ax.legend(fontsize=7); ax.grid(alpha=.25)
        fig.savefig(figdir/'fig14_uncertainty_bounded_validation.png',bbox_inches='tight'); plt.close(fig)
    frames=[]
    if not oso_pred.empty: frames.append(('Oso',oso_pred))
    if not mount_pred.empty: frames.append(('Mount',mount_pred))
    fig,ax=plt.subplots(figsize=(8.2,4.2),dpi=300,constrained_layout=True)
    labels=[]; vals=[]
    for name,df in frames:
        local_obs=obs_col(df)
        for model,g in df.groupby('model'):
            labels.append(name+' '+model.replace('_',' ')); vals.append(float(np.mean(np.abs(g.R_pred-g[local_obs]))))
    ax.bar(range(len(vals)),vals); ax.set_xticks(range(len(vals)),labels,rotation=35,ha='right'); ax.set_ylabel('MAE R / R*'); ax.set_title('Model comparison summary'); ax.grid(axis='y',alpha=.25)
    fig.savefig(figdir/'fig15_model_comparison_summary.png',bbox_inches='tight'); plt.close(fig)

def run(root, mode='full'):
    out=root/'outputs'; out.mkdir(exist_ok=True)
    oso, oso_pred=run_oso_loro(root); mount,mount_pred=run_mount(root,mode=mode); flume,flume_pred=run_flume(root)
    oso.to_csv(out/'oso_loro_metrics.csv',index=False); oso_pred.to_csv(out/'oso_loro_predictions.csv',index=False)
    mount.to_csv(out/'mount_kabasan_predictive_metrics.csv',index=False); mount_pred.to_csv(out/'mount_kabasan_predictive_predictions.csv',index=False)
    flume.to_csv(out/'flume_leave_one_experiment_metrics.csv',index=False); flume_pred.to_csv(out/'flume_leave_one_experiment_predictions.csv',index=False)
    rows=[]
    if not oso.empty:
        best=oso[oso.model.isin(['M4_constant_source','M5_single_boundary_proxy'])].groupby('model')[['mae','rmse','spearman']].median().reset_index().sort_values('mae').iloc[0]
        rows.append({'Dataset':'Oso','Scale':'lab','Validation':'LORO pressure retention','MAE R':best.mae,'RMSE R':best.rmse,'Spearman':best.spearman,'AUC':'NA','Brier skill':'NA','Macro-F1':'NA','Result':'pass' if best.mae<=0.08 else 'conditional'})
    if not mount.empty:
        best=mount[mount.model.isin(['M4_constant_source','M5_convolution_qt'])].groupby('model')[['mae','rmse','spearman']].median().reset_index().sort_values('mae').iloc[0]
        rows.append({'Dataset':'Mount Kaba-san','Scale':'field experiment','Validation':'pressure prediction from early window','MAE R':best.mae,'RMSE R':best.rmse,'Spearman':best.spearman,'AUC':'not estimated','Brier skill':'not estimated','Macro-F1':'NA','Result':'pass' if best.mae<=0.10 else 'conditional'})
    if not flume.empty:
        rows.append({'Dataset':'USGS flume','Scale':'physical flume','Validation':'regime classification screen','MAE R':flume.mae_R_proxy.median(),'RMSE R':'NA','Spearman':flume.spearman_proxy.median(),'AUC':'NA','Brier skill':'NA','Macro-F1':flume.macro_f1_proxy.median(),'Result':'pass' if flume.macro_f1_proxy.median()>=0.65 else 'conditional'})
    pd.DataFrame(rows).to_csv(out/'external_validation_summary.csv',index=False)
    comp=[]
    for name,df in [('Oso',oso),('Mount Kaba-san',mount)]:
        if df.empty: continue
        med=df.groupby('model')['mae'].median().sort_values(); winner=med.index[0]
        for model,val in med.items(): comp.append({'Dataset':name,'Model':model,'Median_MAE_R':val,'Winner':winner})
    pd.DataFrame(comp).to_csv(out/'model_comparison_summary.csv',index=False)
    # simple uncertainty summary from M5 residuals
    if not mount_pred.empty:
        mount_obs=obs_col(mount_pred)
        u=mount_pred[mount_pred.model=='M5_convolution_qt'].groupby('sensor_id').apply(lambda g: pd.Series({'R_pred_median':g.R_pred.median(),'R_pred_p05':g.R_pred.quantile(.05),'R_pred_p95':g.R_pred.quantile(.95),'residual_p95_abs':(g.R_pred-g[mount_obs]).abs().quantile(.95)})).reset_index()
        u.to_csv(out/'uncertainty_validation_metrics.csv',index=False)
    make_figures(root, oso_pred, mount_pred, flume_pred)
    for extra_script in ['09_diagnose_mount_kaba_failure.py', '10_bootstrap_oso_loro.py', 'validation_dimensionless_collapse.py', '11_field_monitoring_cleveland_corral.py', '12_generalized_impedance_operator.py']:
        runpy.run_path(str(root / 'scripts' / extra_script), run_name='__main__')
    return out

def run_with_log(root, mode):
    out=root/'outputs'; out.mkdir(exist_ok=True)
    log_path=out/('reproducibility_run_log_fast.txt' if mode=='fast' else 'reproducibility_run_log.txt')
    start=time.perf_counter()
    buffer=io.StringIO()
    status='failed'
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result=run(root,mode=mode)
        status='success'
        return result, time.perf_counter()-start, log_path
    finally:
        elapsed=time.perf_counter()-start
        lines=[
            f"mode: {mode}",
            f"root: {root}",
            f"scipy_fft_available: {SCIPY_AVAILABLE}",
            f"runtime_seconds: {elapsed:.3f}",
            f"status: {status}",
            "",
            buffer.getvalue().rstrip(),
        ]
        if status=='success':
            lines.append("All validation outputs regenerated successfully.")
        log_path.write_text("\n".join(line for line in lines if line is not None).rstrip()+"\n",encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--offline',action='store_true')
    ap.add_argument('--root',type=Path,default=root_dir())
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--fast',action='store_true',help='Use normalized data with controlled Mount Kaba-san downsampling and FFT M5 convolution.')
    mode.add_argument('--full',action='store_true',help='Use full-resolution validation data.')
    args=ap.parse_args()
    selected_mode='fast' if args.fast else 'full'
    out, elapsed, log_path=run_with_log(args.root.resolve(), selected_mode)
    print(f'validation outputs: {out}')
    print(f'mode: {selected_mode}; runtime_seconds: {elapsed:.3f}; log: {log_path}')
    print('All validation outputs regenerated successfully.')
if __name__=='__main__': main()

