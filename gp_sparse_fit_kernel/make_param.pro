seed = 42
np = 1000
; x = randomu(seed, np)*5
x = rgen(0, 10.0, np)
xt = rgen(-1+min(x), max(x)+1, 200)
yt = cos(xt)
ye = replicate(0.1, np)
y = cos(x) + ye*randomn(seed, np) + 0.2*randomn(seed, np)

mwrfits, /create, {x:x, y:y, ye:ye, xt:xt, yt:yt}, 'input.fits'

plot, x, y, xr=[-1+min(x),max(x)+1], /xs, yr=[-1.2, 1.2], psym=5

end
