The following documents best practices for data access from batch machines. Due to the potentially very high and unpredictable concurrency, users and experiment computing support need to be aware of the limitations imposed by the various storage systems, to avoid overloads and improve their batch job efficiency. Users may need to contact their computing support in order to change non-optimal data placements.

For programs, libraries and readonly input data: make sure that these are stored on a replicated system that is designed to handle massive parallel access.

 

prefer locally-installed programs and libraries (from /usr) over those on shared filesystems (unless your experiment has standardized on particular versions)
your own code will not be installed locally, but if you are missing e.g. a library that is available for the standard software repository but not on LXBATCH, submit a ticket.
in particular, private python installations (such as conda or virtualenv on AFS or EOS) do not scale well.
for CERN- or experiment-specific code and libraries, prefer CVMFS over AFS or EOS
CVMFS also has recent python versions, including many libraries for data analysis and machine learning.
for readonly database access, use a caching system such as FRONTIER
keep temporary files on the LXBATCH nodes (see "DOs" below).
keep result files with intense I/O (rewrites, open+append, many small writes) on LXBATCH nodes - but copy back to stable storage at (successful) end of job.
for experiment and user data, use EOS
in particular, keep ROOT files on EOS - it was developed for these, and can serve them via 'root://'-URLs
for small amounts of data and code, EOS or AFS are alternatives. However, personal AFS "user" and "work" volumes are not replicated, so are not suitable for massive parallel access. EOS can deliver mch higher bandwidth than AFS, but is somewhat slower for metadata-intensive applications. Both are much slower than a local filesystem.
if you are using EOS or AFS, please limit the number of concurrent jobs (by submitting only small batches, or by using max_materialize (see the condor_submit manpage for details). There are no hard rules on how "parallel" you job can be (depends on the I/O pattern), suggest to start with few (10s) and gradually increase.
if you do not really know what your job does with data, running it once on LXPLUS via "strace -tt -e file -f -o $TMPDIR/tracelog .txt -- YOURCOMMAND YOURARGS" (and looking at the resulting logfile $TMPDIR/tracelog .txt - which can be huge) might provide an idea for "unexpected" file accesses.
 
DOs and DON'Ts:
On the batch node, use the job's working directory (or "sandbox") - this resides in a large temporary area under /pool . Please note that /tmp will be "bind mounted" as a subdirectory of the sandbox.
See the (excellent) BATCH documentation on this subject
This directory (with a mostly random name) is automatically created before the jobs starts. It will be the "current working directory" of the job (unless the job itself does "cd" elsewhere) and is suitable for output from the job
This directory gets automatically cleaned up after the job exits. Anything of value in there (such as result files) should be explicitly copied back to a stable shared storage (such as EOS or AFS) at the end of the job
Keep job temporary and stdout+stderr files local on the batch machine, in the per-job pool directory (and - only if needed, i.e while actively debugging - copy them elsewhere at the end of the job). Do not continuously write output log files (e.g. by redirecting output) into a shared storage area such as AFS or EOS.
having the the HTCondor job log (which contains only status updates) on AFS is convenient (allows to use condor_wait), but not required. For massively parallel jobs (several thousands) suggest to not have this file on AFS.
Have output from each batch job go to a separate directory on AFS or EOS. Continuous updates and new files in a single directory on a shared filesystem are particularly bad for caching.
as you cannot have "many" entries (more than a few thousand) in a single directory, you may need some directory tree structure
NEVER "rm -rf" mount points (such as FUSE-mounted EOS). Use the safer "rmdir" instead (after unmounting, and checking return codes), or you might accidentally delete data on the shared storage
Do not implement state machines (list directories, decide on actions, use file existence to track state etc) within each job. Use a database, or use a dedicated submission host.
Use input/output sandboxing (define input and expected output files at job submission time) - HTCondor will ship a local script to the batch node for you, and will copy back STDOUT/STDERR!
also see the EOS File Transfer plugin
Use WLCG "grid"-style job submission whenever possible (besides getting access to many more machines, the job environment is well-defined and does not rely on AFS).
The details vary, but usually can be found in your experiment's computing help pages (look for "Running a grid job")
Use your experiment job frameworks, do not reimplement the wheel. Common tasks such as getting data, storing results, storing (and visualizing) logs and job retry are often already implemented.
EOS-specifics

The fileystem-level access to EOS (FUSE) has less elaborate caching than AFS. Although the limitation from a single AFS volume are gone on EOS, the single namespace still may be a bottleneck - if the application experienced performance problems on AFS, it might continue to do so on EOS.
for high-throughput computing (experiment frameworks, major production/MC activity), we recommend to use application-level access to EOS via "xrdcp" (data copy), "xrdfs" (metadata), or "root://"-URLs (for ROOT) instead of filesystem-level access. The filesystem interface makes it hard to implement client-side timeouts (clients will be "stuck" in kernel space) and retries, and will provide only very reduced error messages (numeric return codes instead of text).
accessing data from many files in a given tree (e.g "grep -r") by a single client will be typically slower on EOS than on AFS (again due to the architecture - the EOS files will be distributed over many servers, and the client needs to look up the location for each file, then contact the corresponding server)
Please see KB0004244
AFS-specifics:

parallel writes to the same AFS volume (e.g. directory) end up on a single RAID-1 or VM blockstorage, and invalidate all caching benefits. Avoid.
Each AFS volume has a small number of server threads available. Exhausting these threads will get your AFS client stuck. Your job will not progress, and eventually HTCondor may terminate it due to too-low CPU efficiency, and the jobs also will get terminated if they affect other AFS users.
AFS replicated readonly volumes can better cope with parallel (read) access, but even at 4 replicas all files in a given volume are only using 8 (spinning) disks. EOS or CEPHFS will typically distribute the files over many more disks. AFS readonly volumes are mostly meant for reliability.
Notes:

"local" storage here refers to a filesystem accessible only by a single machine. Typical examples are /tmp, /pool (for user data), or /usr (for binaries and libraries). For physical machines, these would reside on harddisks (or SSDs) installed into the machine itself. In most cases, this storage is not backed up and may be on unreliable hardware (single disks),  so will be unsuitable for long-term storage, but typically has good performance (since it will only be used by processes within that machine)
"shared" storage refers to data storage provided by some other service such as AFS, NFS, EOS, CephFS etc. The data will be stored typically on redundant hardware and may be (depending on the service) backed up. However, as this storage can be accessed in parallel from many machines, it can be easily overloaded (on a per-file, per-directory or per-"volume" level, depending on the service).