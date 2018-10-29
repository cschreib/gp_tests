seed = 42
xm = 1
np = 5*xm
x = randomu(seed, np)*np
y = cos(x)
xt = rgen(-1, np+1, 200)
yt = cos(xt)
ye = replicate(0.1, np)

mwrfits, /create, {x:x, y:y, ye:ye, xt:xt, yt:yt}, 'input.fits'

plot, x, y, xr=[-1,np+1], /xs, yr=[-1.2, 1.2], psym=5

end
