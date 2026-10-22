/* Optional experiment-local backing for PyTorch POSIX shared tensors.
 * Build: gcc -O2 -Wall -Wextra -shared -fPIC -o torch_shm_redirect.so
 *        torch_shm_redirect.c -ldl
 * Launch only this experiment with LD_PRELOAD and QL_TORCH_SHM_DIR pointing
 * to an existing private (0700) local directory with sufficient free space.
 * Only /torch_* names are redirected; semaphore/other IPC is left untouched.
 * PyTorch still maps shared storage and transfers file descriptors normally.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int torch_path(const char *name, char *path, size_t capacity) {
    const char *root = getenv("QL_TORCH_SHM_DIR");
    if (!root || root[0] != '/' || !name || strncmp(name, "/torch_", 7) != 0
            || strchr(name + 1, '/')) return 0;
    int n = snprintf(path, capacity, "%s/%s", root, name + 1);
    if (n < 0 || (size_t)n >= capacity) { errno = ENAMETOOLONG; return -1; }
    return 1;
}

int shm_open(const char *name, int flags, mode_t mode) {
    char path[PATH_MAX];
    int redirect = torch_path(name, path, sizeof(path));
    if (redirect < 0) return -1;
    if (redirect) return open(path, flags | O_CLOEXEC | O_NOFOLLOW, mode);
    int (*real_open)(const char *, int, mode_t) = dlsym(RTLD_NEXT, "shm_open");
    if (!real_open) { errno = ENOSYS; return -1; }
    return real_open(name, flags, mode);
}

int shm_unlink(const char *name) {
    char path[PATH_MAX];
    int redirect = torch_path(name, path, sizeof(path));
    if (redirect < 0) return -1;
    if (redirect) return unlink(path);
    int (*real_unlink)(const char *) = dlsym(RTLD_NEXT, "shm_unlink");
    if (!real_unlink) { errno = ENOSYS; return -1; }
    return real_unlink(name);
}
