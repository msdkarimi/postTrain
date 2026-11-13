"""Task families for the workspace-agent environment. Importing this package registers every family."""
from envs.tasks.base import FAMILIES, Task, make_task  # noqa: F401
from envs.tasks import fam_csv  # noqa: F401
from envs.tasks import fam_text  # noqa: F401
from envs.tasks import fam_calendar  # noqa: F401
from envs.tasks import fam_files  # noqa: F401
from envs.tasks import fam_api_calendar  # noqa: F401
from envs.tasks import fam_api_comms  # noqa: F401
from envs.tasks import fam_api_files  # noqa: F401
from envs.tasks import fam_more  # noqa: F401
