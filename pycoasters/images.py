import pylab as pl

def show(img,savepath=None):
    dpi = 72.0
    fig = pl.figure(figsize=[x/dpi for x in img.shape[:2]], dpi=dpi)
    pl.figure()
    pl.imshow(img)
    pl.gca().get_xaxis().set_visible(False)
    pl.gca().get_yaxis().set_visible(False)
    if savepath is not None:
        pl.savefig(savepath)
    return fig
