import ROOT
import matplotlib.pyplot as plt

from Logger import *


def set_plot_style_mpl(style="heatmap", n_variables=0):
    """Define a CMS-style matplotlib canvas.

    Args:
        style (str): "heatmap"
        n_variables (int): Used to setup annotation size
    """

    if n_variables < 5: n_variables = 5
    font_size = min(40, max(int(150/n_variables), 7)) if n_variables != 0 else 0
    label_size = min(40, max(int(150/n_variables), 7)) if n_variables != 0 else 15

    if style == "heatmap":
        rc_params = {
            "mathtext.default": "regular",
            "font.size": font_size,
            "axes.labelsize": "large",
            "axes.unicode_minus": False,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": "large",
            "legend.handlelength": 1.5,
            "legend.borderpad": 0.5,
            "legend.frameon": False,
            "xtick.direction": "in",
            "xtick.major.size": 0,
            "xtick.minor.size": 6,
            "xtick.major.pad": 6,
            "xtick.top": True,
            "xtick.major.top": False,
            "xtick.major.bottom": True,
            "xtick.minor.top": False,
            "xtick.minor.bottom": False,
            "xtick.minor.visible": False,
            "xtick.labelsize": label_size,
            "ytick.direction": "in",
            "ytick.major.size": 0.,
            "ytick.minor.size": 6.0,
            "ytick.right": True,
            "ytick.major.left": True,
            "ytick.major.right": False,
            "ytick.minor.left": False,
            "ytick.minor.right": False,
            "ytick.minor.visible": False,
            "ytick.labelsize": label_size,
            "grid.alpha": 0.8,
            "grid.linestyle": ":",
            "axes.linewidth": 2,
            "savefig.transparent": False,
            "figure.figsize": (10.0, 10.0),
            "legend.numpoints": 1,
            "lines.markersize": 8,
        }


    if style == "histogram":
        rc_params = {
            "mathtext.default": "regular",
            "font.size": 25,
            "axes.labelsize": "large",
            "axes.unicode_minus": False,
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
            "legend.fontsize": "large",
            "legend.handlelength": 1.5,
            "legend.borderpad": 0.5,
            "legend.frameon": False,
            "xtick.direction": "in",
            "xtick.major.size": 12,
            "xtick.minor.size": 6,
            "xtick.major.pad": 6,
            "xtick.top": True,
            "xtick.major.top": True,
            "xtick.major.bottom": True,
            "xtick.minor.top": True,
            "xtick.minor.bottom": True,
            "xtick.minor.visible": True,
            "ytick.direction": "in",
            "ytick.major.size": 12,
            "ytick.minor.size": 6.0,
            "ytick.right": True,
            "ytick.major.left": True,
            "ytick.major.right": True,
            "ytick.minor.left": True,
            "ytick.minor.right": True,
            "ytick.minor.visible": True,
            "grid.alpha": 0.8,
            "grid.linestyle": ":",
            "axes.linewidth": 2,
            "savefig.transparent": False,
            "figure.figsize": (15.0, 10.0),
            "legend.numpoints": 1,
            "lines.markersize": 8,
        }
    
    for k, v in rc_params.items():
        plt.rcParams[k] = v


def set_plot_style_root(style="CMS", opt_stat=0, opt_fit=0, font_type=42):

    """Define a CMS-style ROOT canvas.

    Args:
        style (str): "CMS" or "CMS_2D"
        opt_stat (int)
        opt_fit (int)
    """

    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    ROOT.gStyle.SetOptStat(opt_stat)
    ROOT.gStyle.SetOptFit(opt_fit)

    if style.startswith("CMS"):
    
        ROOT.gStyle.SetCanvasBorderMode(0)
        ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
        ROOT.gStyle.SetCanvasDefH(600)
        ROOT.gStyle.SetCanvasDefW(600)
        ROOT.gStyle.SetCanvasDefX(0)
        ROOT.gStyle.SetCanvasDefY(0)
    
        ROOT.gStyle.SetPadTopMargin(0.08)
        ROOT.gStyle.SetPadBottomMargin(0.15)
        ROOT.gStyle.SetPadLeftMargin(0.14)   # 0.13
        if style == "CMS_2D":
            ROOT.gStyle.SetPadRightMargin(0.15)
        else:
            ROOT.gStyle.SetPadRightMargin(0.05)
    
        ROOT.gStyle.SetTextFont(font_type)
        #ROOT.gStyle.SetOptTitle(0)
        #ROOT.gStyle.SetTitleFontSize(0.05)
        #ROOT.gStyle.SetTitleFontSize(20)

        ROOT.gStyle.SetTitleFont(font_type, "XYZ")
        ROOT.gStyle.SetTitleFontSize(18)
        ROOT.gStyle.SetTitleSize(0.06, "XYZ")
        ROOT.gStyle.SetTitleColor(1, "XYZ")
        ROOT.gStyle.SetTitleTextColor(ROOT.kBlack)
        ROOT.gStyle.SetTitleFillColor(10)
        ROOT.gStyle.SetTitleXOffset(1.)
        ROOT.gStyle.SetTitleYOffset(1.2)
    
        ROOT.gStyle.SetLabelColor(1, "XYZ")
        ROOT.gStyle.SetLabelFont(font_type, "XYZ")
        ROOT.gStyle.SetLabelOffset(0.007, "XYZ")
        #ROOT.gStyle.SetLabelSize(25, "XYZ")
        ROOT.gStyle.SetLabelSize(0.04, "XYZ")
    
        ROOT.gStyle.SetAxisColor(1, "XYZ")
        ROOT.gStyle.SetStripDecimals(True)
        ROOT.gStyle.SetTickLength(0.03, "XYZ")
        ROOT.gStyle.SetNdivisions(510, "XYZ")
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
    
        ROOT.gStyle.SetPaperSize(20., 20.)
    
        ROOT.TGaxis.SetExponentOffset(-0.08, 0.01, "Y")

    return


def set_x_label_size(histogram, font_type=3):
    """A default function to get to a correctly sized x-axis label.
    
    The function sets the x-axis label size and returns the size value.

    Args:
        histogram (ROOT.TH1): Histogram 
        font_type (int): The ROOT precision (choose from 2 or 3)

    Returns:
        float
    """

    x_max = histogram.GetXaxis().GetXmax()

    if font_type == 2:
        if x_max > 1000:
            x_label_size = 0.03
        elif x_max > 100:
            x_label_size = 0.04
        else:
            if x_max == 0.0005:
                x_label_size = 0.02
            elif x_max < 0.001:
                x_label_size = 0.022
            elif x_max < 0.005:
                x_label_size = 0.025
            elif x_max == 0.005:
                x_label_size = 0.027
            elif x_max < 0.01:
                x_label_size = 0.032
            elif x_max == 0.05:
                x_label_size = 0.033
            elif x_max < 0.1:
                x_label_size = 0.035
            elif x_max == 0.5:
                x_label_size = 0.04
            elif x_max < 1:
                x_label_size = 0.045
            else:
                x_label_size = 0.045
 
    elif font_type == 3:
        if x_max > 1000:
            x_label_size = 20
        elif x_max > 100:
            x_label_size = 27
        else:
            if x_max == 0.0005:
                x_label_size = 14
            elif x_max < 0.001:
                x_label_size = 15.4
            elif x_max < 0.005:
                x_label_size = 17.5
            elif x_max == 0.005:
                x_label_size = 19
            elif x_max < 0.01:
                x_label_size = 22.4
            elif x_max == 0.05:
                x_label_size = 23
            elif x_max < 0.1:
                x_label_size = 24.5
            elif x_max == 0.5:
                x_label_size = 26
            elif x_max == 1:
                x_label_size = 27
            elif x_max < 1:
                x_label_size = 30
            else:
                x_label_size = 31.5
   
    else:
        log.critical(f"Unknown font type {font_type}!")
        exit(1)

    histogram.GetXaxis().SetLabelSize(x_label_size)

    return x_label_size


def set_x_title_size(histogram, font_type=3):
    """A default function to get to a correctly sized x-axis title.
    
    The function sets the x-axis title size and returns the size value.

    Args:
        histogram (ROOT.TH1): Histogram 
        font_type (int): The ROOT precision (choose from 2 or 3)

    Returns:
        float
    """

    x_label = histogram.GetXaxis().GetTitle()
    x_label_nchar = len(x_label)
    if "#delta" in x_label.lower(): x_label_nchar -= 7
    if "#phi" in x_label.lower(): x_label_nchar -= 7

    if font_type == 2:
        if x_label_nchar > 40:
            x_title_size = 0.035
        else:
            x_title_size = 0.04

    elif font_type == 3:
        if x_label_nchar > 40:
            x_title_size = 24.5
        else:
            x_title_size = 28

    else:
        log.critical(f"Unknown font type {font_type}!")
        exit(1)

    histogram.GetXaxis().SetTitleSize(x_title_size)

    return x_title_size


def set_x_title_offset(histogram):
    """A default function to get to a correctly sized x-axis title.
    
    The function sets the x-axis title size and returns the size value.

    Args:
        histogram (ROOT.TH1): Histogram 
        font_type (int): The ROOT precision (choose from 2 or 3)

    Returns:
        float
    """

    x_label_size = histogram.GetXaxis().GetLabelSize()
    x_title_size = histogram.GetXaxis().GetTitleSize()

    if abs(1 - (x_title_size / x_label_size)) < 0.8:
        x_offset = 1.25
    elif x_title_size > x_label_size:
        x_offset = 1.1
    else:
        x_offset = 1.35

    histogram.GetXaxis().SetTitleOffset(x_offset)

    return x_offset

