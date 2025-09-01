from hypothesis import settings
import warnings

settings.register_profile("default", settings(deadline=None))
warnings.filterwarnings("ignore", category=DeprecationWarning)
