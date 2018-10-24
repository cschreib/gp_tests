npt = 10
seed = 42
x = 2*findgen(npt)/5
; e = replicate(3.0, npt)
e = replicate(1.0, npt)
; y = x + cos(x*2*!dpi/mean(x))
yp = -(x/mean(x)) + 2.5*(x/mean(x))^3
y = yp + e*randomn(seed, npt)

xt = rgen(min(x)-7, max(x)+7, 3000)
yt = -(xt/mean(x)) + 2.5*(xt/mean(x))^3

mwrfits, /create, {x:x, y:y, yp:yp, e:e, xt:xt, yt:yt}, 'data.fits'

nptm = 1000
nmodel = 8

fx = 2*max(x)*findgen(nptm)/nptm - max(x)
fy = fltarr(nptm, nmodel)

for i=0, nmodel-1 do fy[*,i] = (fx/mean(x))^i

prior     = fltarr(nmodel)
prior[1] = -1.5
prior[3] = 0.5

prior_cov = fltarr(nmodel, nmodel)
for i=0, nmodel-1 do prior_cov[i,i] = 1.0

mwrfits, /create, {fx:fx, fy:fy, prior:prior, prior_cov:prior_cov}, 'model.fits'


end
