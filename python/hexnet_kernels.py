"""Opt-in Triton line features, LineConv, masked pooling and masked normalization.

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


@tr.jit(do_not_specialize=['H', 'W', 'S'])
def _line_add(X,Weight,Y,C:tl.constexpr,H,W,S,L:tl.constexpr,K:tl.constexpr):
    plane=tl.program_id(0)
    b,c=plane//C,plane % C
    i=tl.program_id(1)*K+tl.arange(0,K)
    x,y=i % W,i//W
    valid=i<H*W
    at=b*S[0]+c*S[1]+y*S[2]+x*S[3]
    horizontal=tl.full((K,),0.,tl.float32)
    vertical=tl.full((K,),0.,tl.float32)
    diagonal=tl.full((K,),0.,tl.float32)
    for tap in tl.static_range(L):
        d=tap-L//2
        dd=tap-(L-1-L//2)  # centre after the reference diagonal's tap reversal
        wh=tl.load(Weight+(c*3)*L+tap).to(X.dtype.element_ty).to(tl.float32)
        wv=tl.load(Weight+(c*3+1)*L+tap).to(X.dtype.element_ty).to(tl.float32)
        wd=tl.load(Weight+(c*3+2)*L+tap).to(X.dtype.element_ty).to(tl.float32)
        h=tl.load(X+at+d*S[3],valid&(x+d>=0)&(x+d<W),0).to(tl.float32)
        v=tl.load(X+at+d*S[2],valid&(y+d>=0)&(y+d<H),0).to(tl.float32)
        a=tl.load(X+at-dd*S[2]+dd*S[3],valid&(x+dd>=0)&(x+dd<W)&(y-dd>=0)&(y-dd<H),0).to(tl.float32)
        horizontal=tl.fma(h,wh,horizontal)
        vertical=tl.fma(v,wv,vertical)
        diagonal=tl.fma(a,wd,diagonal)
    hv=(horizontal.to(X.dtype.element_ty).to(tl.float32)+vertical).to(X.dtype.element_ty).to(tl.float32)
    line=(hv+diagonal.to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    value=line+tl.load(X+at,valid,0).to(tl.float32)
    tl.store(Y+plane*H*W+i,value,valid)


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
        _line_add[(b*c,tr.cdiv(h*w,256))](x,weight,out,c,h,w,x.stride(),weight.shape[-1],256)
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


# Training kernels take channels-last tensors: element (n, c) of [B, C, H, W] sits at n*C+c for the position
# n = (b*H+y)*W+x, so a tile of positions by channels reads whole rows. Masks and ceilings are [B, 1, H, W].

# Masked pooling: the mean and max of act(x) per sample and channel, read once forward and twice backward, with
# the reference rounding of the bf16 sum, of the mean and max gradients and of their sum. Ties for the max share
# its gradient equally, as amax's does.
@tr.jit(do_not_specialize=['N', 'CS'])
def _pool(X,Ceiling,Count,Out,Top,N,CS,C:tl.constexpr,ACT:tl.constexpr,K:tl.constexpr,CB:tl.constexpr):
    b=tl.program_id(0)
    c=tl.program_id(1)*CB+tl.arange(0,CB)
    total=tl.full((K,CB),0.,tl.float32)
    best=tl.full((K,CB),float('-inf'),tl.float32)
    for start in range(0,N,K):
        j=start+tl.arange(0,K)
        valid=(j<N)[:,None]&(c<C)[None,:]
        v=tl.load(X+(b*N+j)[:,None]*C+c[None,:],valid,0).to(tl.float32)
        if ACT:
            v=tl.minimum(tl.maximum(v,0.),tl.load(Ceiling+b*CS+j,j<N,0).to(tl.float32)[:,None])
        total+=tl.where(valid,v,0.)
        best=tl.maximum(best,tl.where(valid,v,float('-inf')))
    top=tl.max(best,0)
    tl.store(Out+b*2*C+c,tl.sum(total,0).to(X.dtype.element_ty).to(tl.float32)/tl.load(Count+b),c<C)
    tl.store(Out+b*2*C+C+c,top,c<C)
    tl.store(Top+b*C+c,top,c<C)


@tr.jit(do_not_specialize=['N', 'CS'])
def _pool_grad(X,Ceiling,Count,Top,G,DX,N,CS,C:tl.constexpr,ACT:tl.constexpr,K:tl.constexpr,CB:tl.constexpr):
    b=tl.program_id(0)
    c=tl.program_id(1)*CB+tl.arange(0,CB)
    top=tl.load(Top+b*C+c,c<C,0)[None,:]
    ties=tl.full((K,CB),0.,tl.float32)
    for start in range(0,N,K):
        j=start+tl.arange(0,K)
        valid=(j<N)[:,None]&(c<C)[None,:]
        v=tl.load(X+(b*N+j)[:,None]*C+c[None,:],valid,0).to(tl.float32)
        if ACT:
            v=tl.minimum(tl.maximum(v,0.),tl.load(Ceiling+b*CS+j,j<N,0).to(tl.float32)[:,None])
        ties+=tl.where(valid&(v==top),1.,0.)
    mean=(tl.load(G+b*2*C+c,c<C,0)/tl.load(Count+b)).to(DX.dtype.element_ty).to(tl.float32)[None,:]
    peak=tl.load(G+b*2*C+C+c,c<C,0).to(DX.dtype.element_ty).to(tl.float32)
    peak=(peak/tl.sum(ties,0)).to(DX.dtype.element_ty).to(tl.float32)[None,:]
    for start in range(0,N,K):
        j=start+tl.arange(0,K)
        valid=(j<N)[:,None]&(c<C)[None,:]
        at=(b*N+j)[:,None]*C+c[None,:]
        x=tl.load(X+at,valid,0).to(tl.float32)
        v=x
        if ACT:
            ceiling=tl.load(Ceiling+b*CS+j,j<N,0).to(tl.float32)[:,None]
            v=tl.minimum(tl.maximum(x,0.),ceiling)
        g=(mean+tl.where(v==top,peak,0.)).to(DX.dtype.element_ty).to(tl.float32)
        if ACT:
            g=tl.where((x>=0.)&(x<=ceiling),g,0.)
        tl.store(DX+at,g,valid)


class _MaskedPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,count,ceiling):
        b,c,h,w=x.shape
        out=torch.empty((b,2*c),device=x.device,dtype=torch.float32)
        top=torch.empty(b*c,device=x.device,dtype=torch.float32)
        act=ceiling is not None
        source=ceiling if act else x
        _pool[(b,tr.cdiv(c,32))](x,source,count,out,top,h*w,source.stride(0),c,act,32,32)
        ctx.save_for_backward(x,count,top,source)
        ctx.act=act
        return out

    @staticmethod
    def backward(ctx,grad):
        x,count,top,source=ctx.saved_tensors
        b,c,h,w=x.shape
        dx=torch.empty_like(x)
        _pool_grad[(b,tr.cdiv(c,32))](x,source,count,top,grad.contiguous(),dx,h*w,source.stride(0),c,ctx.act,32,32)
        return dx,None,None


def masked_pool(x,count,ceiling=None):
    """hexnet.pool for channels-last CUDA x, count [B, 1] fp32 and an optional ceiling [B, 1, H, W] whose planes are
    contiguous: [B, 2C] fp32."""
    return _MaskedPool.apply(x,count,ceiling)


@tr.jit
def _cells(i, c, N, H, W, MS, C: tl.constexpr):
    """Offsets and bounds of a tile of positions i [K] by channels c [CB], and the mask offsets of its positions."""
    return i[:,None]*C+c[None,:], (i<N)[:,None]&(c<C)[None,:], _offset(i,0,H,W,MS)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'MS', 'T'])
def _reduce(X, M, Mean, Partial, N, H, W, MS, T,
            C: tl.constexpr, P: tl.constexpr, K: tl.constexpr, CB: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1)*CB+tl.arange(0, CB)
    shift = tl.load(Mean+c, c<C, 0).to(X.dtype.element_ty).to(tl.float32)[None,:]
    total = tl.full((K, CB), 0., tl.float32)
    for start in range(t*P, t*P+P, K):
        at, valid, ms = _cells(start+tl.arange(0, K), c, N, H, W, MS, C)
        x = tl.load(X+at, valid, 0).to(tl.float32)
        m = tl.load(M+ms, start+tl.arange(0, K)<N, 0).to(tl.float32)[:,None]
        centred = (x-shift).to(X.dtype.element_ty).to(tl.float32)
        v = ((centred*m).to(X.dtype.element_ty).to(tl.float32)*centred).to(X.dtype.element_ty).to(tl.float32)
        total += tl.where(valid, v, 0)
    tl.store(Partial+c*T+t, tl.sum(total, 0), c<C)


@tr.jit(do_not_specialize=['T'])
def _finish(Partial, Mean, Var, Inv, Cells, X, T,
            EPS: tl.constexpr, K: tl.constexpr):
    c = tl.program_id(0)
    i = tl.arange(0,K)
    value = tl.sum(tl.load(Partial+c*T+i,i<T,0),0)/tl.load(Cells)
    mean = tl.load(Mean+c)
    delta = mean-mean.to(X.dtype.element_ty).to(tl.float32)
    var = tl.maximum(value-delta*delta,0.)
    tl.store(Var+c,var)
    tl.store(Inv+c,tl.rsqrt(var+EPS))


@tr.jit
def _running_update(Mean, Var, RunningMean, RunningVar, Tracked, Cells,
                    C: tl.constexpr, MOM: tl.constexpr, K: tl.constexpr):
    c = tl.arange(0, K)
    valid = c < C
    mean = tl.load(Mean+c, valid, 0)
    var = tl.load(Var+c, valid, 0)
    old_mean = tl.load(RunningMean+c, valid, 0)
    old_var = tl.load(RunningVar+c, valid, 0)
    cells = tl.load(Cells)
    unbiased = tl.div_rn(var*cells, tl.maximum(cells-1., 1.))
    # Match ATen's float32 lerp branch for scalar weights (native/Lerp.h).
    if MOM < .5:
        new_mean = old_mean + MOM*(mean-old_mean)
        new_var = old_var + MOM*(unbiased-old_var)
    else:
        new_mean = mean-(mean-old_mean)*(1.-MOM)
        new_var = unbiased-(unbiased-old_var)*(1.-MOM)
    tl.store(RunningMean+c, new_mean, valid)
    tl.store(RunningVar+c, new_var, valid)
    tl.store(Tracked, tl.load(Tracked)+1)


def norm_update(norm, mean, var, cells):
    """Update fused training norm buffers in one launch; caller skips checkpoint replay."""
    _running_update[(1,)](mean, var, norm.running_mean, norm.running_var,
                          norm.num_batches_tracked, cells, mean.numel(), norm.momentum,
                          tr.next_power_of_2(mean.numel()), enable_fp_fusion=False)


@tr.jit
def _normalized(x,mean,inv,weight,bias,D:tl.constexpr):
    shift=mean.to(D).to(tl.float32)
    centred=(x-shift).to(D).to(tl.float32)
    scale=weight*inv
    offset=(bias-(mean-shift)*scale).to(D).to(tl.float32)
    return tl.fma(centred,scale.to(D).to(tl.float32),offset).to(D).to(tl.float32)


@tr.jit
def _xhat(x,mean,inv,D:tl.constexpr):
    shift=mean.to(D).to(tl.float32)
    centred=(x-shift).to(D).to(tl.float32)
    bias=((shift-mean)*inv).to(D).to(tl.float32)
    return tl.fma(centred,inv.to(D).to(tl.float32),bias).to(D).to(tl.float32)


@tr.jit
def _channels(Mean,Inv,Weight,Bias,c,C:tl.constexpr):
    """Per-channel mean, inverse deviation, weight and bias of the channels c, as [1, CB] rows."""
    return (tl.load(Mean+c,c<C,0)[None,:],tl.load(Inv+c,c<C,0)[None,:],tl.load(Weight+c,c<C,0)[None,:],
            tl.load(Bias+c,c<C,0)[None,:])


@tr.jit(do_not_specialize=['N', 'H', 'W', 'MS'])
def _apply(X,M,Y,Mean,Inv,Weight,Bias,N,H,W,MS,
           ACT:tl.constexpr,C:tl.constexpr,K:tl.constexpr,CB:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    c=tl.program_id(1)*CB+tl.arange(0,CB)
    at,valid,ms=_cells(i,c,N,H,W,MS,C)
    mean,inv,weight,bias=_channels(Mean,Inv,Weight,Bias,c,C)
    y=_normalized(tl.load(X+at,valid,0).to(tl.float32),mean,inv,weight,bias,X.dtype.element_ty)
    if ACT:
        m=tl.load(M+ms,i<N,0)[:,None]
        y=tl.where(m>0,tl.maximum(y,0.),0.)
    tl.store(Y+at,y,valid)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'MS'])
def _grad_products(X,M,G,Gated,Product,Mean,Inv,Weight,Bias,N,H,W,MS,
                   ACT:tl.constexpr,C:tl.constexpr,K:tl.constexpr,CB:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    c=tl.program_id(1)*CB+tl.arange(0,CB)
    at,valid,ms=_cells(i,c,N,H,W,MS,C)
    mean,inv,weight,bias=_channels(Mean,Inv,Weight,Bias,c,C)
    x=tl.load(X+at,valid,0).to(tl.float32)
    g=tl.load(G+at,valid,0).to(tl.float32)
    if ACT:
        m=tl.load(M+ms,i<N,0)[:,None]
        y=_normalized(x,mean,inv,weight,bias,X.dtype.element_ty)
        g=tl.where((y>=0)&((m>0)|(y<=0)),g,0.)
    g=g.to(Gated.dtype.element_ty).to(tl.float32)
    tl.store(Gated+at,g,valid)
    tl.store(Product+at,(g*_xhat(x,mean,inv,X.dtype.element_ty)).to(Product.dtype.element_ty),valid)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'MS'])
def _grad_apply(X,M,G,Dx,Mean,Inv,Weight,Bias,Db,Dw,Cells,N,H,W,MS,
                C:tl.constexpr,K:tl.constexpr,CB:tl.constexpr):
    i=tl.program_id(0)*K+tl.arange(0,K)
    c=tl.program_id(1)*CB+tl.arange(0,CB)
    at,valid,ms=_cells(i,c,N,H,W,MS,C)
    mean,inv,weight,_=_channels(Mean,Inv,Weight,Bias,c,C)
    x=tl.load(X+at,valid,0).to(tl.float32)
    m=tl.load(M+ms,i<N,0).to(tl.float32)[:,None]
    g=tl.load(G+at,valid,0).to(tl.float32)
    h=_xhat(x,mean,inv,X.dtype.element_ty)
    cells=tl.load(Cells)
    a=(tl.load(Db+c,c<C,0)/cells).to(X.dtype.element_ty).to(tl.float32)[None,:]
    b=(tl.load(Dw+c,c<C,0)/cells).to(X.dtype.element_ty).to(tl.float32)[None,:]
    correction=tl.fma(h,b,a).to(X.dtype.element_ty).to(tl.float32)
    dx=(g-(m*correction).to(X.dtype.element_ty).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    dx=dx*(weight*inv).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Dx+at,dx,valid)


class MaskedBatchNorm(torch.autograd.Function):
    """hexnet._MaskedBatchNorm for channels-last CUDA x and a [B, 1, H, W] mask, with the activation act(y, ceiling) fused when `activate`."""
    @staticmethod
    def forward(ctx,x,mask,weight,bias,cells,eps,activate=False):
        b,c,h,w=x.shape
        n,p=b*h*w,1024
        t=tr.cdiv(n,p)
        partial=torch.empty((c,t),device=x.device,dtype=torch.float32)
        # Match ATen's reduction order. Sub-ULP mean differences can change
        # BF16 activations enough to exceed the full-model gradient tolerance.
        mean=(x*mask).sum((0,2,3),dtype=torch.float32)/cells
        var=torch.empty_like(weight)
        inv=torch.empty_like(weight)
        y=torch.empty_like(x)
        ms=mask.stride()
        _reduce[(t,tr.cdiv(c,32))](x,mask,mean,partial,n,h,w,ms,t,c,p,64,32,enable_fp_fusion=False)
        _finish[(c,)](partial,mean,var,inv,cells,x,t,eps,tr.next_power_of_2(t),enable_fp_fusion=False)
        _apply[(tr.cdiv(n,64),tr.cdiv(c,32))](x,mask,y,mean,inv,weight,bias,n,h,w,ms,activate,c,64,32,enable_fp_fusion=False)
        ctx.save_for_backward(x,mask,weight,bias,mean,inv,cells)
        ctx.activate=activate
        return y,mean,var

    @staticmethod
    def backward(ctx,grad,_mean,_var):
        x,mask,weight,bias,mean,inv,cells=ctx.saved_tensors
        b,c,h,w=x.shape
        n=b*h*w
        grad=grad.contiguous(memory_format=torch.channels_last)
        gated=torch.empty_like(grad)
        product=torch.empty_like(grad,dtype=torch.promote_types(grad.dtype,x.dtype))
        dx=torch.empty_like(x)
        ms=mask.stride()
        grid=(tr.cdiv(n,64),tr.cdiv(c,32))
        _grad_products[grid](x,mask,grad,gated,product,mean,inv,weight,bias,n,h,w,ms,ctx.activate,c,64,32,enable_fp_fusion=False)
        # Preserve the reference reduction order before rounding the correction
        # coefficients to bf16. A different tree can change those coefficients.
        db=gated.sum((0,2,3),dtype=mean.dtype)
        dw=product.sum((0,2,3),dtype=mean.dtype)
        _grad_apply[grid](x,mask,gated,dx,mean,inv,weight,bias,db,dw,cells,n,h,w,ms,c,64,32,enable_fp_fusion=False)
        return (dx,None,dw.to(weight.dtype),db.to(weight.dtype),None,None,None)[:len(ctx.needs_input_grad)]


# Training LineConv: the forward pass is _line_add_nhwc. The backward pass runs the transposed taps over the gradient
# with the reference bf16 rounding of each axis sum, and reduces the tap gradients in fp32.
@tr.jit(do_not_specialize=['N', 'H', 'W'])
def _line_grad(G,Weight,DX,N,C:tl.constexpr,H,W,L:tl.constexpr,P:tl.constexpr,CH:tl.constexpr):
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
        wh=tl.load(Weight+(c*3)*L+tap,c<C,0).to(G.dtype.element_ty).to(tl.float32)
        wv=tl.load(Weight+(c*3+1)*L+tap,c<C,0).to(G.dtype.element_ty).to(tl.float32)
        wd=tl.load(Weight+(c*3+2)*L+tap,c<C,0).to(G.dtype.element_ty).to(tl.float32)
        h=tl.load(G+at-d*C,valid&((x-d>=0)&(x-d<W))[:,None],0).to(tl.float32)
        v=tl.load(G+at-d*W*C,valid&((y-d>=0)&(y-d<H))[:,None],0).to(tl.float32)
        a=tl.load(G+at+dd*(W-1)*C,valid&((x-dd>=0)&(x-dd<W)&(y+dd>=0)&(y+dd<H))[:,None],0).to(tl.float32)
        horizontal=tl.fma(h,wh[None,:],horizontal)
        vertical=tl.fma(v,wv[None,:],vertical)
        diagonal=tl.fma(a,wd[None,:],diagonal)
    hv=(horizontal.to(G.dtype.element_ty).to(tl.float32)+vertical.to(G.dtype.element_ty).to(tl.float32))
    hv=hv.to(G.dtype.element_ty).to(tl.float32)
    residual=(hv+tl.load(G+at,valid,0).to(tl.float32)).to(G.dtype.element_ty).to(tl.float32)
    tl.store(DX+at,residual+diagonal.to(G.dtype.element_ty).to(tl.float32),valid)


@tr.jit(do_not_specialize=['N', 'H', 'W', 'T'])
def _line_taps(X,G,Partial,N,C:tl.constexpr,H,W,T,P:tl.constexpr,L:tl.constexpr,LP:tl.constexpr,
               K:tl.constexpr,CH:tl.constexpr):
    c=tl.program_id(0)*CH+tl.arange(0,CH)
    t=tl.program_id(1)
    taps=tl.arange(0,LP)[:,None]
    horizontal=tl.full((LP,CH),0.,tl.float32)
    vertical=tl.full((LP,CH),0.,tl.float32)
    diagonal=tl.full((LP,CH),0.,tl.float32)
    for start in range(t*P,t*P+P,K):
        p=start+tl.arange(0,K)
        x,y=p % W,p//W % H
        at=p[:,None]*C+c[None,:]
        valid=(p[:,None]<N)&(c[None,:]<C)
        g=tl.load(G+at,valid,0).to(tl.float32)
        for tap in tl.static_range(L):
            d=tap-L//2
            dd=tap-(L-1-L//2)
            h=tl.load(X+at+d*C,valid&((x+d>=0)&(x+d<W))[:,None],0).to(tl.float32)
            v=tl.load(X+at+d*W*C,valid&((y+d>=0)&(y+d<H))[:,None],0).to(tl.float32)
            a=tl.load(X+at+dd*(1-W)*C,valid&((x+dd>=0)&(x+dd<W)&(y-dd>=0)&(y-dd<H))[:,None],0).to(tl.float32)
            row=taps==tap
            horizontal+=tl.where(row,tl.sum(g*h,0)[None,:],0.)
            vertical+=tl.where(row,tl.sum(g*v,0)[None,:],0.)
            diagonal+=tl.where(row,tl.sum(g*a,0)[None,:],0.)
    out=Partial+(c[None,:]*3*L+taps)*T+t
    used=(taps<L)&(c[None,:]<C)
    tl.store(out,horizontal,used)
    tl.store(out+L*T,vertical,used)
    tl.store(out+2*L*T,diagonal,used)


class _TrainLineAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        ctx.save_for_backward(x,weight)
        return line_add(x,weight)

    @staticmethod
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        b,c,h,w=x.shape
        n,length=b*h*w,weight.shape[-1]
        grad=grad.to(x.dtype).contiguous(memory_format=torch.channels_last)
        dx=dw=None
        if ctx.needs_input_grad[0]:
            dx=torch.empty_like(grad)
            _line_grad[(tr.cdiv(n,16),tr.cdiv(c,32))](grad,weight,dx,n,c,h,w,length,16,32)
        if ctx.needs_input_grad[1]:
            p=4096
            t=tr.cdiv(n,p)
            partial=torch.empty((c,3,length,t),dtype=torch.float32,device=grad.device)
            _line_taps[(tr.cdiv(c,32),t)](x,grad,partial,n,c,h,w,t,p,length,tr.next_power_of_2(length),32,32)
            dw=partial.sum(-1)
        return dx,dw


def line_train_add(x,weight):
    """Residual LineConv x + LineConv(x) for channels-last CUDA bf16 activations and fp32 taps, with gradients."""
    return _TrainLineAdd.apply(x,weight)
