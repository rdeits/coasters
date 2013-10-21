import pylab as pl

def show(img,savepath=None):
    dpi = 72.0
    pl.figure(figsize=[x/dpi for x in img.shape[:2]], dpi=dpi)
    pl.figure()
    pl.imshow(img)
    if savepath is not None:
        pl.savefig(savepath)
