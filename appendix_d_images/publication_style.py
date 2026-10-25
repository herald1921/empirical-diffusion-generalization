"""Visual roles are scoped by panel; these constants never transform data."""
BLUE = '#0072B2'
ORANGE = '#D55E00'
TEAL = '#008578'
INK = '#263742'
GRAY = '#7A8994'
LIGHT = '#E4E9ED'
METHOD_COLORS = {'empirical': BLUE, 'mlp': GRAY, 'unet': TEAL}
DIRECTION_STYLE = {'emp': ('-', 'o'), 'nn': ('--', 's')}
N_COLORS = {100: '#318AC3', 1000: '#4F6CC1', 10000: '#7352AD'}
FIELD_PAIRS = [
 ('mlp_unet', 'MLP vs U-Net', TEAL, '-'),
 ('empirical_mlp', 'Empirical vs MLP', '#82929F', '--'),
 ('empirical_unet', 'Empirical vs U-Net', '#C17E47', '-.')]

def apply(plt):
 plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
  'mathtext.fontset':'stix','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,
  'xtick.labelsize':6.7,'ytick.labelsize':6.7,'legend.fontsize':7,
  'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,
  'axes.edgecolor':'#98A7B1','axes.spines.top':False,'axes.spines.right':False,
  'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
