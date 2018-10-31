seed = 42
np = 100
; x = randomu(seed, np)*5
x = rgen(0, 10.0, np)
xt = rgen(-1+min(x), max(x)+1, 200)
yt = (cos(xt) + 3*exp(-4*(xt - 4)^2))*abs(1+xt)
ye = replicate(0.3, np)
y = (cos(x) + 3*exp(-4*(x - 4)^2))*abs(1+x) + ye*randomn(seed, np) + 1.0*randomn(seed, np)

scale = 1.0

; id = where(x lt 3 or x gt 6)
id = where(finite(x))
x = x[id]
y = y[id]*scale
ye = ye[id]*scale
yt *= scale

mwrfits, /create, {x:x, y:y, ye:ye, xt:xt, yt:yt}, 'input.fits'

plot, x, y, xr=[-1+min(x),max(x)+1], /xs, yr=[-10.2*scale, 10.2*scale], psym=5

end
