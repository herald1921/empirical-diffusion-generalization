# -*- coding: utf-8 -*-
"""Projected-neighborhood diversity deficit; no generated-law or OT rerun.

Frozen N=16 designs, existing 512-point empirical benchmarks, and unchanged
population laws. Ellipse/sphere values are undefined outside their reach tube.
The segment uses its globally unique convex projection. No figure-eight value.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
TAU = 2 * np.pi
DELTA = .05
N = 16
RHO_ELLIPSE = .5
R_SPHERE = 2.

def c_delta(dim):
    return np.sqrt(dim) + np.sqrt(2*np.log(1/DELTA))

def ellipse_point(theta):
    return np.array([2*np.cos(theta), np.sin(theta)])

def ellipse_arc(site, radius):
    center = float(np.mod(np.arctan2(site[1],site[0]/2), TAU))
    if radius == 0:return center, center
    def tangent_distance(theta):
        y = ellipse_point(theta)
        tangent = np.array([-2*np.sin(theta),np.cos(theta)])
        return abs(float(np.dot(site-y,tangent)))/np.linalg.norm(tangent)
    ends=[]
    for sign in [-1,1]:
        # First root on each side bounds the connected projected image of a ball.
        step=.002
        while tangent_distance(center+sign*step)<radius:
            step*=1.4
            if step>np.pi:raise RuntimeError('No local arc endpoint found')
        offset=brentq(lambda d:tangent_distance(center+sign*d)-radius,0,step,xtol=4e-15)
        theta=center+sign*offset
        y=ellipse_point(theta)
        normal=np.array([np.cos(theta)/2,np.sin(theta)])
        normal/=np.linalg.norm(normal)
        normal_coordinate=float(np.dot(site-y,normal))
        assert abs(normal_coordinate)<RHO_ELLIPSE+1e-10
        assert abs(tangent_distance(theta)-radius)<2e-12
        ends.append(theta)
    return tuple(ends)

def ellipse_intervals(training,radius):
    intervals=[]
    for site in training:
        lo,hi=ellipse_arc(site,radius)
        if lo<0:intervals.extend([(0.,hi),(lo+TAU,TAU)])
        elif hi>TAU:intervals.extend([(lo,TAU),(0.,hi-TAU)])
        else:intervals.append((lo,hi))
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]+2e-14:merged[-1][1]=max(merged[-1][1],hi)
        else:merged.append([lo,hi])
    gaps=[];last=0.
    for lo,hi in merged:
        if lo>last+2e-14:gaps.append((last,lo))
        last=hi
    if last<TAU-2e-14:gaps.append((last,TAU))
    endpoints=np.array([ellipse_point(t) for interval in merged for t in interval])
    return merged,gaps,endpoints

def ellipse_squared(training,radius,tol=2e-12):
    if not 0<=radius<RHO_ELLIPSE:return float('nan'),float('nan')
    _,gaps,endpoints=ellipse_intervals(training,radius)
    if not gaps:return 0.,0.
    def integrand(theta):
        point=ellipse_point(theta)
        distance2=np.min(np.sum((endpoints-point)**2,axis=1))
        return distance2*(1+.1*np.cos(theta))/TAU
    total=0.;error=0.
    for lo,hi in gaps:
        # Interior stationary nearest points would make a normal chord >=2 reach,
        # while a training site is always within <0.437. Endpoints therefore suffice.
        value,err=quad(integrand,lo,hi,epsabs=tol,epsrel=tol,limit=160)
        total+=value;error+=err
    return max(total,0.),error

def ellipse_mesh_check(training,radius,power=18):
    intervals,_,endpoints=ellipse_intervals(training,radius)
    theta=(np.arange(2**power)+.5)*TAU/(2**power)
    covered=np.zeros(theta.shape,dtype=bool)
    for lo,hi in intervals:covered|=(theta>=lo)&(theta<=hi)
    miss=theta[~covered]
    points=np.column_stack([2*np.cos(miss),np.sin(miss)])
    dist2=np.full(len(miss),np.inf)
    for endpoint in endpoints:dist2=np.minimum(dist2,np.sum((points-endpoint)**2,axis=1))
    return float(np.sum(dist2*(1+.1*np.cos(miss)))/(2**power))

def project_ellipse_direct(points):
    """Independent Lagrange-multiplier nearest-point calculation."""
    axes2=np.array([4.,1.]);value=np.abs(points)
    off=value[:,1]>1e-14;answer=np.zeros_like(points)
    if off.any():
        values=value[off];lo=np.full(len(values),np.nextafter(-1.,0.));hi=np.maximum(1.,2*np.linalg.norm(values,axis=1))
        for _ in range(85):
            middle=(lo+hi)/2
            residual=np.sum(axes2*values**2/(axes2+middle[:,None])**2,axis=1)-1
            lo=np.where(residual>0,middle,lo);hi=np.where(residual>0,hi,middle)
        answer[off]=axes2*values/(axes2+((lo+hi)/2)[:,None])
    if (~off).any():
        px=np.minimum(2.,4*value[~off,0]/3)
        answer[~off]=np.column_stack([px,np.sqrt(np.maximum(0.,1-px**2/4))])
    return answer*np.where(points<0,-1.,1.)

def ellipse_projection_checks(training):
    report=[]
    phase=TAU*np.arange(1024)/1024
    directions=np.column_stack([np.cos(phase),np.sin(phase)])
    for r in [.08,.24,.49]:
        endpoint_errors=[];outside=[]
        for site in training:
            center=float(np.mod(np.arctan2(site[1],site[0]/2),TAU))
            lo,hi=ellipse_arc(site,r)
            projected=project_ellipse_direct(site+r*directions)
            angles=np.arctan2(projected[:,1],projected[:,0]/2)
            angles=center+(angles-center+np.pi)%TAU-np.pi
            outside.append(max(lo-float(angles.min()),float(angles.max())-hi,0.))
            for theta in [lo,hi]:
                point=ellipse_point(theta)
                normal=np.array([np.cos(theta)/2,np.sin(theta)]);normal/=np.linalg.norm(normal)
                normal_foot=point+np.dot(site-point,normal)*normal
                got=project_ellipse_direct(normal_foot[None])[0]
                endpoint_errors.append(float(np.linalg.norm(got-point)))
                endpoint_errors.append(abs(float(np.linalg.norm(normal_foot-site))-r))
        report.append(dict(radius=r,boundary_directions_per_site=1024,maximum_arc_violation=max(outside),maximum_endpoint_projection_or_ball_error=max(endpoint_errors)))
    assert max(x['maximum_arc_violation'] for x in report)<1e-10
    assert max(x['maximum_endpoint_projection_or_ball_error'] for x in report)<1e-10
    return report

def sphere_angles(training,power,seed):
    uv=qmc.Sobol(2,scramble=True,seed=seed).random_base2(power)
    z=1-2*uv[:,0];theta=TAU*uv[:,1];radial=np.sqrt(1-z*z)
    targets=np.column_stack([radial*np.cos(theta),radial*np.sin(theta),z])
    nearest_dot=np.full(len(targets),-1.)
    for site in training/R_SPHERE:
        nearest_dot=np.maximum(nearest_dot,targets@site)
    return np.sort(np.arccos(np.clip(nearest_dot,-1,1)))

def sphere_cover(training):
    hull=ConvexHull(training/R_SPHERE)
    distances=-hull.equations[:,-1]
    assert np.min(distances)>0
    angle=float(np.arccos(np.min(distances)))
    return angle,float(R_SPHERE*np.sin(angle))

def sphere_series(training,radii,power=20,seed=777):
    angles=sphere_angles(training,power,seed)
    cover_angle,cover_radius=sphere_cover(training)
    # Descending cumulative sums allow an exact average over the chosen quadrature.
    reverse=angles[::-1]
    cosine=np.r_[0.,np.cumsum(np.cos(reverse))]
    sine=np.r_[0.,np.cumsum(np.sin(reverse))]
    out=[]
    for r in radii:
        if not 0<=r<R_SPHERE:out.append(float('nan'));continue
        if r>=cover_radius:out.append(0.);continue
        beta=np.arcsin(r/R_SPHERE)
        start=np.searchsorted(angles,beta,side='right');count=len(angles)-start
        val=2*R_SPHERE**2*(count-cosine[count]*np.cos(beta)-sine[count]*np.sin(beta))/len(angles)
        if val<1e-7:
            # Stable near exact coverage; avoid cancellation in 1-cos(small).
            val=float(4*R_SPHERE**2*np.sum(np.sin((angles[start:]-beta)/2)**2)/len(angles))
        out.append(max(float(val),0.))
    return np.asarray(out),{'power':power,'seed':seed,'cover_angle':cover_angle,'cover_radius':cover_radius,'largest_sample_angle':float(angles[-1])}

def segment_squared(radius):
    half_cell=np.sqrt(3)/N
    return np.maximum(half_cell-np.asarray(radius),0.)**3/(3*half_cell)

def main():
    started=time.time()
    parser=argparse.ArgumentParser();parser.add_argument('--sphere-power',type=int,default=20)
    parser.add_argument('--repository',type=Path,help='Optional original repository root for frozen-input hash and baseline verification; default uses bundled inputs.')
    args=parser.parse_args()
    meta=json.loads((ROOT/'inputs/source_metadata.json').read_text())
    if args.repository is not None:
        for entry in meta['designs'].values():
            original=args.repository/entry['original_path']
            assert hashlib.sha256(original.read_bytes()).hexdigest()==entry['sha256']
        original_audit=json.loads((args.repository/meta['original_audit_path']).read_text())
        assert {k:v['endpoint_SN'] for k,v in original_audit['supports'].items()}==meta['endpoint_SN']
    ellipse=np.load(ROOT/'inputs/ellipse_training.npy');sphere=np.load(ROOT/'inputs/sphere_training.npy')
    for name,values in [('ellipse',ellipse),('sphere',sphere)]:assert values.shape==(16,2 if name=='ellipse' else 3)
    sphere_cover_angle,sphere_cover_radius=sphere_cover(sphere)
    ordinary=np.geomspace(.008,2.,401)
    special=[0.,*meta['original_eta_grid'],RHO_ELLIPSE/c_delta(2)*(1-1e-10),RHO_ELLIPSE/c_delta(2),R_SPHERE/c_delta(3)*(1-1e-10),R_SPHERE/c_delta(3),sphere_cover_radius/c_delta(3),np.sqrt(3)/N/c_delta(2)]
    eta=np.unique(np.r_[ordinary,special]);radii2=eta*c_delta(2);radii3=eta*c_delta(3)
    e=np.array([ellipse_squared(ellipse,r) for r in radii2]);sphere_values,sphere_info=sphere_series(sphere,radii3,args.sphere_power,777)
    segment=segment_squared(radii2)
    data={'ellipse':e[:,0],'sphere':sphere_values,'segment':segment}
    rows=[]
    for shape,values in data.items():
        radius=radii3 if shape=='sphere' else radii2
        for noise,r,value in zip(eta,radius,values):
            g=float(np.sqrt(value));sn=meta['endpoint_SN'][shape]
            rows.append(dict(shape=shape,eta=float(noise),r=float(r),G=g,G_over_SN=g/sn,defined=int(np.isfinite(value)),exact_zero=int(np.isfinite(value) and value==0)))
    with (ROOT/'diversity_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Independent grid/refinement checks at informative scales, including r=0.
    ellipse_checks=[]
    for r in [0.,.02,.08,.16,.24,.32,.40,.49]:
        fine,err=ellipse_squared(ellipse,r,1e-14);base,_=ellipse_squared(ellipse,r);mesh=ellipse_mesh_check(ellipse,r)
        ellipse_checks.append(dict(radius=r,G_squared=fine,adaptive_error_estimate=err,base_squared_difference=abs(fine-base),mesh_squared_difference=abs(fine-mesh)))
    sphere_radii=np.array([0.,.08,.2,.4,.7,1.,sphere_cover_radius*.90,sphere_cover_radius*.99,sphere_cover_radius,1.9])
    main_check,info=sphere_series(sphere,sphere_radii,args.sphere_power,777)
    coarse,_=sphere_series(sphere,sphere_radii,max(16,args.sphere_power-2),777)
    independent,_=sphere_series(sphere,sphere_radii,args.sphere_power,20261105)
    sphere_checks=[dict(radius=float(r),G_squared=float(v),coarse_squared_difference=float(abs(v-c)),independent_squared_difference=float(abs(v-i)),independent_G_difference=float(abs(np.sqrt(v)-np.sqrt(i)))) for r,v,c,i in zip(sphere_radii,main_check,coarse,independent)]
    # Exact segment formula also checked against direct one-dimensional quadrature.
    half=np.sqrt(3)/N
    segment_checks=[]
    for r in [0.,.02,.07,.10,half,.2]:
        val=quad(lambda x:max(abs(x)-r,0.)**2/(2*half),-half,half,points=[-min(r,half),min(r,half)],epsabs=1e-15)[0]
        segment_checks.append(dict(radius=r,formula=float(segment_squared(r)),integral=val,difference=abs(val-segment_squared(r))))
    theta=np.sort(np.mod(np.arctan2(ellipse[:,1],ellipse[:,0]/2),TAU))
    gaps=np.diff(np.r_[theta,theta[0]+TAU]);cover_bound=float(gaps.max())
    assert cover_bound<2*RHO_ELLIPSE
    for shape,values in data.items():
        finite=values[np.isfinite(values)];assert np.max(np.diff(finite))<1e-10
    report={'definition':'G(r)=inf_{supp(v) subset Pi(U_r)} W2(p,v), ambient cost; projected-neighborhood deficit, not a denoised-law component measured on full support','normalization':'Use existing 512-point assignment endpoint_SN from frozen Figure 8; do not force G(0)=S_N.','input_provenance':meta,'eta_grid_count':len(eta),'eta_plot_range':[.008,2.],'radius_calibration':'r=(sqrt(D)+sqrt(2 log(20))) eta','domains':{'ellipse':{'reach':.5,'eta_strict_upper':float(.5/c_delta(2))},'sphere':{'reach':2.,'eta_strict_upper':float(2/c_delta(3))},'segment':{'reach':'infinity','definition_extension':'same projected-neighborhood formula using globally unique projection onto the convex segment'},'eight':{'defined':False,'reason':'self-intersection gives zero reach; omitted per user decision'}},'formulas':{'sphere':'A_r is union of spherical caps of angular radius beta=asin(r/2); G^2=8 E[1-cos((theta_nearest-beta)_+)].','segment':'Let a=sqrt(3), l=a/16. G^2=((l-r)_+)^3/(3l).','ellipse':'Pi(ball(z_i,r)) is connected parameter arc; endpoints solve |(z_i-y(theta)) dot unit_tangent(theta)|=r. Distance to uncovered points is minimum distance to arc endpoints, integrated against (1+.1 cos(theta))/(2pi).'},'ellipse_sample_cover_upper_bound':cover_bound,'ellipse_max_adaptive_error_estimate':float(np.nanmax(e[:,1])),'ellipse_checks':ellipse_checks,'sphere':sphere_info,'sphere_checks':sphere_checks,'segment_checks':segment_checks,'zero_on_log_axis':'Exact zero values are retained in CSV. They must not be represented as positive values without marking a display floor; no line outside defined domains.','limitations':'Sphere values are numerical Sobol quadrature, with refinement and independent-scramble checks, not certified deterministic continuum error bounds. Existing W2/WT/WN curves and empirical normalization were not recomputed.','runtime_seconds':time.time()-started}
    report['ellipse_projection_checks']=ellipse_projection_checks(ellipse)
    report['normalization_discretization_note']='The frozen 512-point assignment SN is an existing numerical benchmark, not the exact continuum W2. In particular segment G(0)/SN=1.0004886391691556 because the frozen finite benchmark is slightly below the continuum value 1/16; do not alter SN or force endpoint equality.'
    (ROOT/'diversity_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'eta_count':len(eta),'ellipse_mesh_squared_max':max(x['mesh_squared_difference'] for x in ellipse_checks),'sphere_independent_G_max':max(x['independent_G_difference'] for x in sphere_checks),'sphere_cover_eta':sphere_cover_radius/c_delta(3),'runtime_seconds':report['runtime_seconds']},indent=2))

if __name__=='__main__':main()
