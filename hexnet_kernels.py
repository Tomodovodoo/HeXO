"""Opt-in Triton line features, inference LineConv, and masked normalization.

Keep the reference bf16 rounding points, including the centred variance and
the products used for the affine gradients. Reductions accumulate in fp32.
Batch, canvas and strides are runtime values, with scalar specialization disabled.
Only model constants, layout and tile sizes specialize the actor kernels.
"""
import os
from pathlib import Path

# One persistent per-user cache, independent of actor PID and working directory.
# Respect an explicit shared cache supplied by the launcher.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path.home()/".triton"/"cache"))

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=['N', 'H', 'W', 'OS', 'PS', 'MS'])
def _windows(Own,Opp,Mask,Counts,N,H,W,
             OS,PS,MS,K:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    axis=tl.program_id(1)
    b,y,x=i//(H*W),i//W % H,i % W
    dx=tl.where(axis==1,0,1)
    dy=tl.where(axis==0,0,tl.where(axis==1,1,-1))
    own=tl.full((K,),0,tl.float32)
    opp=tl.full((K,),0,tl.float32)
    mask=tl.full((K,),0,tl.float32)
    for tap in range(6):
        yy,xx=y+tap*dy,x+tap*dx
        valid=(i<N)&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)
        own+=tl.load(Own+b*OS[0]+yy*OS[2]+xx*OS[3],valid,0).to(Counts.dtype.element_ty).to(tl.float32)
        opp+=tl.load(Opp+b*PS[0]+yy*PS[2]+xx*PS[3],valid,0).to(Counts.dtype.element_ty).to(tl.float32)
        mask+=tl.load(Mask+b*MS[0]+yy*MS[2]+xx*MS[3],valid,0).to(Counts.dtype.element_ty).to(tl.float32)
    own=own.to(Counts.dtype.element_ty).to(tl.float32)
    opp=opp.to(Counts.dtype.element_ty).to(tl.float32)
    mask=mask.to(Counts.dtype.element_ty).to(tl.float32)
    at=(b*6+axis)*H*W+y*W+x
    tl.store(Counts+at,tl.where((mask>5.5)&(opp<.5),own,0.),i<N)
    tl.store(Counts+at+3*H*W,tl.where((mask>5.5)&(own<.5),opp,0.),i<N)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'OS', 'PS', 'MS'])
def _features(Own,Opp,Mask,Counts,Out,N,H,W,
              OS,PS,MS,K:tl.constexpr,NHWC:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    b,y,x=i//(H*W),i//W % H,i % W
    mine=tl.full((K,),0.,tl.float32)
    theirs=tl.full((K,),0.,tl.float32)
    for axis in tl.static_range(3):
        dx=0 if axis==1 else 1
        dy=0 if axis==0 else 1 if axis==1 else -1
        own=tl.full((K,),0.,tl.float32)
        opp=tl.full((K,),0.,tl.float32)
        for tap in range(6):
            yy,xx=y-tap*dy,x-tap*dx
            valid=(i<N)&(yy>=0)&(yy<H)&(xx>=0)&(xx<W)
            at=(b*6+axis)*H*W+yy*W+xx
            own=tl.maximum(own,tl.load(Counts+at,valid,0).to(tl.float32))
            opp=tl.maximum(opp,tl.load(Counts+at+3*H*W,valid,0).to(tl.float32))
        at=(b*H*W+y*W+x)*10+axis if NHWC else (b*10+axis)*H*W+y*W+x
        cs=1 if NHWC else H*W
        tl.store(Out+at,(own/6.).to(Counts.dtype.element_ty),i<N)
        tl.store(Out+at+3*cs,(opp/6.).to(Counts.dtype.element_ty),i<N)
        mine=tl.maximum(mine,own)
        theirs=tl.maximum(theirs,opp)
    own=tl.load(Own+b*OS[0]+y*OS[2]+x*OS[3],i<N,0).to(tl.float32)
    opp=tl.load(Opp+b*PS[0]+y*PS[2]+x*PS[3],i<N,0).to(tl.float32)
    mask=tl.load(Mask+b*MS[0]+y*MS[2]+x*MS[3],i<N,0).to(tl.float32)
    empty=mask*((1.-own).to(Own.dtype.element_ty).to(tl.float32)-opp).to(Own.dtype.element_ty).to(tl.float32)
    at=(b*H*W+y*W+x)*10+6 if NHWC else (b*10+6)*H*W+y*W+x
    cs=1 if NHWC else H*W
    tl.store(Out+at,tl.where(mine>3.5,empty,0.),i<N)
    tl.store(Out+at+cs,tl.where(theirs>3.5,empty,0.),i<N)
    tl.store(Out+at+2*cs,tl.where(mine>4.5,empty,0.),i<N)
    tl.store(Out+at+3*cs,tl.where(theirs>4.5,empty,0.),i<N)


def line_features(own,opp,mask):
    b,_,h,w=own.shape
    dtype=torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else own.dtype
    counts=torch.empty((b,6,h,w),device=own.device,dtype=dtype)
    nhwc=own.stride(1)==1
    out=torch.empty((b,10,h,w),device=own.device,dtype=torch.promote_types(dtype,own.dtype),
                    memory_format=torch.channels_last if nhwc else torch.contiguous_format)
    args=(b*h*w,h,w,own.stride(),opp.stride(),mask.stride(),256)
    _windows[(tr.cdiv(b*h*w,256),3)](own,opp,mask,counts,*args,enable_fp_fusion=False)
    _features[(tr.cdiv(b*h*w,256),)](own,opp,mask,counts,out,*args,nhwc,enable_fp_fusion=False)
    return out


@tr.jit(do_not_specialize=['B', 'H', 'W', 'S'])
def _line_add(X,Weight,Y,B,C:tl.constexpr,H,W,
              S,L:tl.constexpr,K:tl.constexpr,NHWC:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    if NHWC:
        c,x,y,b=i % C,i//C % W,i//(C*W) % H,i//(C*H*W)
    else:
        x,y=i % W,i//W % H
        c,b=i//(H*W) % C,i//(C*H*W)
    valid=i<B*C*H*W
    at=b*S[0]+c*S[1]+y*S[2]+x*S[3]
    horizontal=tl.full((K,),0.,tl.float32)
    vertical=tl.full((K,),0.,tl.float32)
    diagonal=tl.full((K,),0.,tl.float32)
    for tap in tl.static_range(L):
        d=tap-L//2
        dd=tap-(L-1-L//2)  # centre after the reference diagonal's tap reversal
        wh=tl.load(Weight+(c*3)*L+tap,valid,0).to(X.dtype.element_ty).to(tl.float32)
        wv=tl.load(Weight+(c*3+1)*L+tap,valid,0).to(X.dtype.element_ty).to(tl.float32)
        wd=tl.load(Weight+(c*3+2)*L+tap,valid,0).to(X.dtype.element_ty).to(tl.float32)
        h=tl.load(X+at+d*S[3],valid&(x+d>=0)&(x+d<W),0).to(tl.float32)
        v=tl.load(X+at+d*S[2],valid&(y+d>=0)&(y+d<H),0).to(tl.float32)
        a=tl.load(X+at-dd*S[2]+dd*S[3],valid&(x+dd>=0)&(x+dd<W)&(y-dd>=0)&(y-dd<H),0).to(tl.float32)
        horizontal=tl.fma(h,wh,horizontal)
        vertical=tl.fma(v,wv,vertical)
        diagonal=tl.fma(a,wd,diagonal)
    hv=(horizontal.to(X.dtype.element_ty).to(tl.float32)+vertical).to(X.dtype.element_ty).to(tl.float32)
    line=(hv+diagonal.to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    value=line+tl.load(X+at,valid,0).to(tl.float32)
    tl.store(Y+i,value,valid)


@tr.jit(do_not_specialize=['N', 'H', 'W'])
def _line_add_nhwc(X,Weight,Y,N,C:tl.constexpr,H,W,
                   L:tl.constexpr,P:tl.constexpr,CH:tl.constexpr):
    p=tl.program_id(0)*P+tl.arange(0,P)
    c=tl.program_id(1)*CH+tl.arange(0,CH)
    x,y=p % W,p//W % H
    at=p[:,None]*C+c[None,:]
    valid=(p[:,None]<N)&(c[None,:]<C)
    horizontal=tl.full((P,CH),0.,tl.float32)
    vertical=tl.full((P,CH),0.,tl.float32)
    diagonal=tl.full((P,CH),0.,tl.float32)
    for tap in tl.static_range(L):
        d=tap-L//2
        dd=tap-(L-1-L//2)
        wh=tl.load(Weight+(c*3)*L+tap,c<C,0).to(X.dtype.element_ty).to(tl.float32)
        wv=tl.load(Weight+(c*3+1)*L+tap,c<C,0).to(X.dtype.element_ty).to(tl.float32)
        wd=tl.load(Weight+(c*3+2)*L+tap,c<C,0).to(X.dtype.element_ty).to(tl.float32)
        h=tl.load(X+at+d*C,valid&((x+d>=0)&(x+d<W))[:,None],0).to(tl.float32)
        v=tl.load(X+at+d*W*C,valid&((y+d>=0)&(y+d<H))[:,None],0).to(tl.float32)
        a=tl.load(X+at+dd*(1-W)*C,valid&((x+dd>=0)&(x+dd<W)&(y-dd>=0)&(y-dd<H))[:,None],0).to(tl.float32)
        horizontal=tl.fma(h,wh[None,:],horizontal)
        vertical=tl.fma(v,wv[None,:],vertical)
        diagonal=tl.fma(a,wd[None,:],diagonal)
    hv=(horizontal.to(X.dtype.element_ty).to(tl.float32)+vertical).to(X.dtype.element_ty).to(tl.float32)
    line=(hv+diagonal.to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y+at,line+tl.load(X+at,valid,0).to(tl.float32),valid)


def line_add(x,weight):
    out=torch.empty_like(x)
    b,c,h,w=x.shape
    if x.is_contiguous(memory_format=torch.channels_last):
        _line_add_nhwc[(tr.cdiv(b*h*w,16),tr.cdiv(c,32))](x,weight,out,b*h*w,c,h,w,weight.shape[-1],16,32)
    else:
        _line_add[(tr.cdiv(x.numel(),256),)](x,weight,out,*x.shape,x.stride(),weight.shape[-1],256,False)
    return out


@tr.jit(do_not_specialize=['H', 'W', 'S'])
def _offset(i, c, H, W, S):
    return i//(H*W)*S[0]+c*S[1]+(i//W % H)*S[2]+(i % W)*S[3]


@tr.jit(do_not_specialize=['N', 'H', 'W', 'XS', 'MS', 'YS'])
def _eval(X,M,Y,Mean,Var,Weight,Bias,N,H,W,
          XS,MS,YS,EPS:tl.constexpr,K:tl.constexpr,C:tl.constexpr,NHWC:tl.constexpr):
    if NHWC:
        at=tl.program_id(0)*K+tl.arange(0,K)
        c,i=at % C,at//C
    else:
        c=tl.program_id(0)
        i=tl.program_id(1)*K+tl.arange(0,K)
    x=tl.load(X+_offset(i,c,H,W,XS),i<N,0).to(tl.float32)
    m=tl.load(M+_offset(i,c,H,W,MS),i<N,0)
    y=((x-tl.load(Mean+c))*tl.rsqrt(tl.load(Var+c)+EPS))*tl.load(Weight+c)+tl.load(Bias+c)
    y=y.to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y+_offset(i,c,H,W,YS),tl.where(m>0,tl.maximum(y,0.),0.),i<N)


def norm_eval(norm,x,mask):
    b,c,h,w=x.shape
    out=torch.empty_like(x)
    ms=mask.stride() if mask.shape[1]==c else (mask.stride(0),0,*mask.stride()[2:])
    grid=(tr.cdiv(x.numel(),1024),) if x.stride(1)==1 else (c,tr.cdiv(b*h*w,1024))
    _eval[grid](x,mask,out,norm.running_mean,norm.running_var,norm.weight,norm.bias,
                b*h*w,h,w,x.stride(),ms,out.stride(),norm.eps,1024,c,x.stride(1)==1,enable_fp_fusion=False)
    return out


@tr.jit(do_not_specialize=['N', 'H', 'W', 'XS', 'MS', 'T'])
def _reduce(X, M, Mean, Partial, N, H, W,
            XS, MS, T, VAR: tl.constexpr,
            K: tl.constexpr):
    c, t = tl.program_id(0), tl.program_id(1)
    i = t*K+tl.arange(0, K)
    x = tl.load(X+_offset(i,c,H,W,XS), i<N, 0).to(tl.float32)
    m = tl.load(M+_offset(i,c,H,W,MS), i<N, 0).to(tl.float32)
    if VAR:
        shift = tl.load(Mean+c).to(X.dtype.element_ty).to(tl.float32)
        centred = (x-shift).to(X.dtype.element_ty).to(tl.float32)
        v = ((centred*m).to(X.dtype.element_ty).to(tl.float32)*centred).to(X.dtype.element_ty).to(tl.float32)
    else:
        v = (x*m).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Partial+c*T+t, tl.sum(tl.where(i<N,v,0),0))


@tr.jit(do_not_specialize=['T'])
def _finish(Partial, Mean, Var, Inv, Cells, X, T,
            EPS: tl.constexpr, VAR: tl.constexpr, K: tl.constexpr):
    c = tl.program_id(0)
    i = tl.arange(0,K)
    value = tl.sum(tl.load(Partial+c*T+i,i<T,0),0)/tl.load(Cells)
    if VAR:
        mean = tl.load(Mean+c)
        delta = mean-mean.to(X.dtype.element_ty).to(tl.float32)
        var = tl.maximum(value-delta*delta,0.)
        tl.store(Var+c,var)
        tl.store(Inv+c,tl.rsqrt(var+EPS))
    else:
        tl.store(Mean+c,value)


@tr.jit
def _normalized(x,mean,inv,weight,bias,D:tl.constexpr):
    shift=mean.to(D).to(tl.float32)
    centred=(x-shift).to(D).to(tl.float32)
    scale=weight*inv
    offset=(bias-(mean-shift)*scale).to(D).to(tl.float32)
    return tl.fma(centred,scale.to(D).to(tl.float32),offset).to(D).to(tl.float32)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'XS', 'MS', 'YS'])
def _apply(X,M,Y,Mean,Inv,Weight,Bias,N,H,W,
           XS,MS,YS,ACT:tl.constexpr,K:tl.constexpr):
    c=tl.program_id(0)
    i=tl.program_id(1)*K+tl.arange(0,K)
    x=tl.load(X+_offset(i,c,H,W,XS),i<N,0).to(tl.float32)
    y=_normalized(x,tl.load(Mean+c),tl.load(Inv+c),tl.load(Weight+c),tl.load(Bias+c),X.dtype.element_ty)
    if ACT:
        m=tl.load(M+_offset(i,c,H,W,MS),i<N,0)
        y=tl.where(m>0,tl.maximum(y,0.),0.)
    tl.store(Y+_offset(i,c,H,W,YS),y,i<N)


@tr.jit
def _xhat(x,mean,inv,D:tl.constexpr):
    shift=mean.to(D).to(tl.float32)
    centred=(x-shift).to(D).to(tl.float32)
    bias=((shift-mean)*inv).to(D).to(tl.float32)
    return tl.fma(centred,inv.to(D).to(tl.float32),bias).to(D).to(tl.float32)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'XS', 'MS', 'GS', 'T'])
def _grad_reduce(X,M,G,Mean,Inv,Weight,Bias,Partial,N,H,W,
                 XS,MS,GS,C:tl.constexpr,T,ACT:tl.constexpr,K:tl.constexpr):
    c,t=tl.program_id(0),tl.program_id(1)
    i=t*K+tl.arange(0,K)
    x=tl.load(X+_offset(i,c,H,W,XS),i<N,0).to(tl.float32)
    g=tl.load(G+_offset(i,c,H,W,GS),i<N,0).to(tl.float32)
    if ACT:
        m=tl.load(M+_offset(i,c,H,W,MS),i<N,0)
        y=_normalized(x,tl.load(Mean+c),tl.load(Inv+c),tl.load(Weight+c),tl.load(Bias+c),X.dtype.element_ty)
        g=tl.where((y>=0)&((m>0)|(y<=0)),g,0.)
    h=_xhat(x,tl.load(Mean+c),tl.load(Inv+c),X.dtype.element_ty)
    dw=(g*h).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Partial+c*T+t,tl.sum(g,0))
    tl.store(Partial+(C+c)*T+t,tl.sum(tl.where(i<N,dw,0),0))


@tr.jit(do_not_specialize=['T'])
def _grad_finish(Partial,Db,Dw,C:tl.constexpr,T,K:tl.constexpr):
    c=tl.program_id(0)
    i=tl.arange(0,K)
    tl.store(Db+c,tl.sum(tl.load(Partial+c*T+i,i<T,0),0))
    tl.store(Dw+c,tl.sum(tl.load(Partial+(C+c)*T+i,i<T,0),0))


@tr.jit(do_not_specialize=['N', 'H', 'W', 'XS', 'MS', 'GS', 'DS'])
def _grad_apply(X,M,G,Dx,Mean,Inv,Weight,Bias,Db,Dw,Cells,N,H,W,
                XS,MS,GS,DS,ACT:tl.constexpr,K:tl.constexpr):
    c=tl.program_id(0)
    i=tl.program_id(1)*K+tl.arange(0,K)
    x=tl.load(X+_offset(i,c,H,W,XS),i<N,0).to(tl.float32)
    m=tl.load(M+_offset(i,c,H,W,MS),i<N,0).to(tl.float32)
    g=tl.load(G+_offset(i,c,H,W,GS),i<N,0).to(tl.float32)
    inv=tl.load(Inv+c)
    if ACT:
        y=_normalized(x,tl.load(Mean+c),inv,tl.load(Weight+c),tl.load(Bias+c),X.dtype.element_ty)
        g=tl.where((y>=0)&((m>0)|(y<=0)),g,0.)
    h=_xhat(x,tl.load(Mean+c),inv,X.dtype.element_ty)
    cells=tl.load(Cells)
    a=(tl.load(Db+c)/cells).to(X.dtype.element_ty).to(tl.float32)
    b=(tl.load(Dw+c)/cells).to(X.dtype.element_ty).to(tl.float32)
    correction=tl.fma(h,b,a).to(X.dtype.element_ty).to(tl.float32)
    dx=(g-(m*correction).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    dx=dx*(tl.load(Weight+c)*inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Dx+_offset(i,c,H,W,DS),dx,i<N)


class MaskedBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mask,weight,bias,cells,eps,activate=False):
        b,c,h,w=x.shape
        n,k=b*h*w,1024
        t=tr.cdiv(n,k)
        partial=torch.empty((c,t),device=x.device,dtype=torch.float32)
        mean=torch.empty_like(weight)
        var=torch.empty_like(weight)
        inv=torch.empty_like(weight)
        y=torch.empty_like(x)
        ms=mask.stride() if mask.shape[1]==c else (mask.stride(0),0,*mask.stride()[2:])
        grid=(c,t)
        common=dict(N=n,H=h,W=w,XS=x.stride(),MS=ms,T=t,K=k,enable_fp_fusion=False)
        _reduce[grid](x,mask,mean,partial,VAR=False,**common)
        _finish[(c,)](partial,mean,var,inv,cells,x,t,eps,False,tr.next_power_of_2(t),enable_fp_fusion=False)
        _reduce[grid](x,mask,mean,partial,VAR=True,**common)
        _finish[(c,)](partial,mean,var,inv,cells,x,t,eps,True,tr.next_power_of_2(t),enable_fp_fusion=False)
        _apply[grid](x,mask,y,mean,inv,weight,bias,n,h,w,x.stride(),ms,y.stride(),activate,k,enable_fp_fusion=False)
        ctx.save_for_backward(x,mask,weight,bias,mean,inv,cells)
        ctx.activate=activate
        return y,mean,var

    @staticmethod
    def backward(ctx,grad,_mean,_var):
        x,mask,weight,bias,mean,inv,cells=ctx.saved_tensors
        b,c,h,w=x.shape
        n,k=b*h*w,1024
        t=tr.cdiv(n,k)
        partial=torch.empty((2,c,t),device=x.device,dtype=torch.float32)
        db,dw=torch.empty_like(weight),torch.empty_like(weight)
        dx=torch.empty_like(x)
        ms=mask.stride() if mask.shape[1]==c else (mask.stride(0),0,*mask.stride()[2:])
        _grad_reduce[(c,t)](x,mask,grad,mean,inv,weight,bias,partial,n,h,w,x.stride(),ms,grad.stride(),c,t,ctx.activate,k,enable_fp_fusion=False)
        _grad_finish[(c,)](partial,db,dw,c,t,tr.next_power_of_2(t),enable_fp_fusion=False)
        _grad_apply[(c,t)](x,mask,grad,dx,mean,inv,weight,bias,db,dw,cells,n,h,w,
                            x.stride(),ms,grad.stride(),dx.stride(),ctx.activate,k,enable_fp_fusion=False)
        return (dx,None,dw,db,None,None,None)[:len(ctx.needs_input_grad)]
