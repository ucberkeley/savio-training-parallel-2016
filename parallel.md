% Savio parallelization training: Parallelized usage of the Berkeley Savio high-performance computing cluster
% September 27, 2016

# Status

 - Chris will be working through this first draft to finalize existing content Wed/Thu
 - major gaps
     - JupyterHub worked example
     - ht-helper example
 - looking for feedback
     - examples of using other build systems
     - examples of things that can arise when managing dependencies amongst user-installed software (any things other than setting PATH or LD_LIBRARY_PATH?)
     - parallelization strategies
     - real-world Python or R examples would be nice but could also be overly involved


# Introduction

We'll do this mostly as a demonstration. I encourage you to login to your account and try out the various examples yourself as we go through them.

Some of this material is based on the extensive Savio documention we have prepared and continue to prepare, available at [http://research-it.berkeley.edu/services/high-performance-computing/user-guide](http://research-it.berkeley.edu/services/high-performance-computing/user-guide).

The materials for this tutorial are available using git at [https://github.com/ucberkeley/savio-training-parallel-2016](https://github.com/ucberkeley/savio-training-parallel-2016) or simply as a [zip file](https://github.com/ucberkeley/savio-training-parallel-2016/archive/master.zip).

Please see this [zip file](https://github.com/ucberkeley/savio-training-parallel-2016/archive/master.zip) for materials from our introductory training on August 2, including accessing Savio, data transfer, and basic job submission.


# Outline

This training session will cover the following topics:

 - Software installation
     - Installing third-party software
     - Build systems other than autoconf+make?
     - Installing Python and R packages that rely on third-party software
         - Python example?
         - R example
 - Parallelization strategies
     - Some general principles and concepts    
         - shared vs. distributed memory; communication overhead
         - hybrid and nested parallelization
         - load-balancing and prescheduling          
     - Overview of software tools
 - Setting up a parallel job in SLURM
     - Job submission overview
     - SLURM flags
     - SLURM environment variables
 - Basic parallelization in Python and R
     - iPython examples
         - JupyterHub
         - threaded linear algebra
     - R examples
 - High-throughput computing with ht_helper
     - example [HELP!!]  
 - Wrap-up


# Software installation - third-party software

In general, third-party software will provide installation instructions on a webpage, Github README, or install file inside the package source code.

The key for installing on Savio is making sure everything gets installed in your own home/project/scratch directory and making sure you have the packages on which the software depends on also installed or loaded from the Savio modules. 

A common installation approach is the GNU build system (Autotools), which involves three steps: configure, make, and make install.

  - configure: this queries your system to find out what tools (e.g., compilers and other packages) you have available to use in building and using the software
  - make: this compiles the source code in the software
  - make install: this moves the compiled code (library files) and include files and the like to their permanent home 

Here's an example of you might install a piece of software in your home directory

```
mkdir software; cd software
mkdir src; cd src  # set up a directory for source packages
# install gdal, needed for rgdal
V=2.0.2
PKG=gdal
INSTALLDIR=~/software/${PKG}
wget http://download.osgeo.org/${PKG}/${PKG}-${V}.tar.bz2
bunzip2 ${PKG}-${V}.bz2
cd ${PKG}-${V}
./configure --prefix=$INSTALLDIR | tee ../configure.log   # --prefix is key to install in directory you have access to
make | tee ../make.log
make install | tee ../install.log
```

For Cmake, the following may work:
```
$PKG=foo
cmake -DCMAKE_INSTALL_PREFIX=/global/home/users/$USER/software/$PKG ..
```

If you're then going to install additional software that uses the software you just installed and that software needs to link against compiled code from the installed software, you may need something like this:

```
# needed in the geos example to install the rgeos R package
export LD_LIBRARY_PATH=${INSTALLDIR}/lib:${LD_LIBRARY_PATH}
```

This is because Linux only looks in certain directories for the location of .so library files.

In other cases you might need to add the location of an executable to your PATH variable so that the operating system can find the executable. Linux only looks in certain directories for executables.

```
# not needed in the geos example
# export PATH=${INSTALLDIR}/bin:${PATH}
```


# Installing Python and R packages 

If you see comments about `libfoo.so` not found, see above comment about modifying your LD_LIBRARY_PATH environment variable. 

```
module load python/3.5
module load cuda
PKG=pycuda
pip install --user ${PKG}
ls .local
```

[[ perhaps python yaml pkg that depends on libyaml.so ]]

```
module load R
Rscript -e "install.packages('rgeos', repos = 'http://cran.cnr.berkeley.edu', lib = Sys.getenv('R_LIB_DIR'))"
TMP=`R CMD config R_LIB_DIR`
echo ${TMP}
ls ${TMP}
```

# Parallel processing terminology

  - *cores*: We'll use this term to mean the different processing
units available on a single node.
  - *nodes*: We'll use this term to mean the different computers,
each with their own distinct memory, that make up a cluster or supercomputer.
  - *processes* or *SLURM tasks*: computational instances executing on a machine; multiple
processes may be executing at once. A given program may start up multiple
processes at once. Ideally we have no more processes than cores on
a node.
  - *threads*: multiple paths of execution within a single process;
the OS sees the threads as a single process, but one can think of
them as 'lightweight' processes. Ideally when considering the processes
and their threads, we would have no more processes and threads combined
than cores on a node.
 - *computational tasks*: We'll use this to mean the independent computational units that make up the job you submit
    - each *process* or *SLURM task* might carry out one computational task or might be assigned multiple tasks sequentially or as a group.

# Parallelization strategies

The following are some basic principles/suggestions for how to parallelize
your computation.

[UNDER CONSTRUCTION - feedback welcome (looking primarily for content feedback at this point)]

Should I use one machine/node or many machines/nodes?

 - If you can do your computation on the cores of a single node using
shared memory, that will be faster than using the same number of cores
(or even somewhat more cores) across multiple nodes. Similarly, jobs
with a lot of data/high memory requirements that one might think of
as requiring Spark or Hadoop may in some cases be much faster if you can find
a single machine with a lot of memory.
 - That said, if you would run out of memory on a single node, then you'll
need to use distributed memory.
 - If you have so much data that you overwhelm the amount that can fit in RAM on one machine, Spark may be useful.
 - If you have data that will fit in memory on one machine, Python, MATLAB, C/C++, and R may be your best bet.

What level or dimension should I parallelize over?

 - If you have nested loops, you often only want to parallelize at
one level of the code. Keep in mind whether your linear algebra is being
threaded. Often you will want to parallelize over a loop and not use
threaded linear algebra.
 - Often it makes sense to parallelize the outer loop when you have nested
loops.
 - You generally want to parallelize in such a way that your code is
load-balanced and does not involve too much communication. 

 - If you have a small-ish number of long task, then a hybrid parallelization scheme may make sense.
 - E.g., if each task involves substantial linear algebra, you might have multiple cores on a node assigned to each task so that the linear algebra can be done in parallel.

How do I balance communication overhead with keeping my cores busy?

 - If you have very few tasks, particularly if the tasks take different
amounts of time, often some of the processors will be idle and your code
poorly load-balanced.
 - If you have very many tasks and each one takes little time, the communication
overhead of starting and stopping the tasks will reduce efficiency.
 - Avoid having a very small number of jobs, each of which (or some of which) take hours to days to run
 - Avoid having a very large number of jobs, each of which takes milliseconds to run

Should multiple tasks be pre-assigned to a process (i.e., a worker) (sometimes called *prescheduling*) or should tasks
be assigned dynamically as previous tasks finish? 

 - Basically if you have many tasks that each take similar time, you
want to preschedule the tasks to reduce communication. If you have few tasks
or tasks with highly variable completion times, you don't want to
preschedule, to improve load-balancing.
 - For R in particular, some of R's parallel functions allow you to say whether the 
tasks should be prescheduled. E.g., `library(Rmpi); help(mpi.parSapply)` gives some information.
 - Or you may want to manually aggregate your tasks if each one is very quick.

# Parallelization tools

 - shared memory parallelization (one machine, multiple cores)
    - threaded linear algebra in R, Python, MATLAB 
        -(for R and Python, they need to be installed with parallel linear algebra support from OpenBLAS or MKL)
    - parallelization of independent computations
        - iPython (example below) or other Python packages (e.g., `pp`, `multiprocessing`)
        - various R packages (foreach + doParallel, mclapply, parLapply)
        - parfor in MATLAB
    - openMP for writing threaded code in C/C++
    - GPUs: various machine learning packages with GPU back-end support, direct coding in CUDA or openCL

 - distributed parallelization (multiple machines (nodes))
    - parallelization of independent computations
        - iPython
        - various R packages (foreach + doMPI, foreach + doSNOW, pbdR)
        - parfor in MATLAB with MATLAB DCS
    - MPI for more tightly-coupled parallelization
        - MPI in C/C++
        - mpi4py for Python
        - pbdR (pbdMPI) and Rmpi for R
    - Spark/Hadoop for parallelized MapReduce computations across multiple nodes
        - data spread across multiple nodes and read into collective memory


# Submitting jobs: accounts and partitions

All computations are done by submitting jobs to the scheduling software that manages jobs on the cluster, called SLURM.

When submitting a job, the main things you need to indicate are the project account you are using (in some cases you might have access to multiple accounts such as an FCA and a condo) and the partition.

You can see what accounts you have access to and which partitions within those accounts as follows:

```
sacctmgr -p show associations user=SAVIO_USERNAME
```

Here's an example of the output for a user who has access to an FCA, a condo, and a special partner account:
```
Cluster|Account|User|Partition|Share|GrpJobs|GrpTRES|GrpSubmit|GrpWall|GrpTRESMins|MaxJobs|MaxTRES|MaxTRESPerNode|MaxSubmit|MaxWall|MaxTRESMins|QOS|Def QOS|GrpTRESRunMins|
brc|co_stat|paciorek|savio2_gpu|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio2_htc|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio_bigmem|1||||||||||||savio_lowprio|savio_lowprio||
brc|co_stat|paciorek|savio2|1||||||||||||savio_lowprio,stat_normal|stat_normal||
brc|fc_paciorek|paciorek|savio2|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|fc_paciorek|paciorek|savio_bigmem|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|ac_scsguest|paciorek|savio2_htc|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|ac_scsguest|paciorek|savio2_gpu|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|ac_scsguest|paciorek|savio2|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|ac_scsguest|paciorek|savio_bigmem|1||||||||||||savio_debug,savio_normal|savio_normal||
brc|ac_scsguest|paciorek|savio|1||||||||||||savio_debug,savio_normal|savio_normal||
```

If you are part of a condo, you'll notice that you have *low-priority* access to certain partitions. For example I am part of the statistics cluster *co_stat*, which owns some Savio2 nodes and therefore I have normal access to those, but I can also burst beyond the condo and use other partitions at low-priority (see below).

In contrast, through my FCA, I have access to the savio, savio2, and big memory partitions.

# Submitting a batch job

Let's see how to submit a simple job. If your job will only use the resources on a single node, you can do the following. 


Here's an example job script that I'll run. You'll need to modify the --account value and possibly the --partition value.


        #!/bin/bash
        # Job name:
        #SBATCH --job-name=test
        #
        # Account:
        #SBATCH --account=co_stat
        #
        # Partition:
        #SBATCH --partition=savio2
        #
        # Wall clock limit (30 seconds here):
        #SBATCH --time=00:00:30
        #
        ## Command(s) to run:
        module load python/3.2.3 numpy
        python3 calc.py >& calc.out


Now let's submit and monitor the job:

```
sbatch job.sh

squeue -j JOB_ID

wwall -j JOB_ID
```

Note that except for the *savio2_htc*  and *savio2_gpu* partitions, all jobs are given exclusive access to the entire node or nodes assigned to the job (and your account is charged for all of the cores on the node(s). 

# Parallel job submission

If you are submitting a job that uses multiple nodes, you'll need to carefully specify the resources you need. The key flags for use in your job script are:

 - `--nodes` (or `-N`): indicates the number of nodes to use
 - `--ntasks-per-node`: indicates the number of tasks (i.e., processes) one wants to run on each node
 - `--cpus-per-task` (or `-c`): indicates the number of cpus to be used for each task

In addition, in some cases it can make sense to use the `--ntasks` (or `-n`) option to indicate the total number of tasks and let the scheduler determine how many nodes and tasks per node are needed. In general `--cpus-per-task` will be 1 except when running threaded code.  


Here's an example job script for a job that uses MPI for parallelizing over multiple nodes:

       #!/bin/bash
       # Job name:
       #SBATCH --job-name=test
       #
       # Account:
       #SBATCH --account=account_name
       #
       # Partition:
       #SBATCH --partition=partition_name
       #
       # Number of MPI tasks needed for use case (example):
       #SBATCH --ntasks=40
       #
       # Processors per task:
       #SBATCH --cpus-per-task=1
       #
       # Wall clock limit:
       #SBATCH --time=00:00:30
       #
       ## Command(s) to run (example):
       module load intel openmpi
       mpirun ./a.out



Some common paradigms are:

 - MPI jobs that use *one* CPU per task for each of *n* tasks
     - `--ntasks=n --cpus-per-task=1` 
     - `--nodes=x --ntasks-per-node=y --cpus-per-task=1` (assuming that `n = x*y`)
 - openMP/threaded jobs that use *c* CPUs for *one* task
     - `--nodes=1 --ntasks-per-node=1 --cpus-per-task=c` 
 - hybrid parallelization jobs (e.g., MPI+threading) that use *c* CPUs for each of *n* tasks
     - `--ntasks=n --cpus-per-task=c`
     - `--nodes=x --ntasks-per-node=y cpus-per-task=c` (assuming that `y*c` equals the number of cores on a node and that `n = x*y` equals the total number of tasks

In general, the defaults for the various flags will be 1 so some of the flags above are not strictly needed.

There are lots more examples of job submission scripts for different kinds of parallelization (multi-node (MPI), multi-core (openMP), hybrid, etc.) [here](http://research-it.berkeley.edu/services/high-performance-computing/running-your-jobs#Job-submission-with-specific-resource-requirements). We'll discuss some of them below.

# SLURM environment variables

When you write your code, you may need to specify information in your code about the number of cores to use. SLURM will provide a variety of variables that you can use in your code so that it adapts to the resources you have requested rather than being hard-coded. 

Here are some of the variables that may be useful: SLURM_NTASKS, SLURM_CPUS_PER_TASK, SLURM_NODELIST, SLURM_NNODES.

Here's how you can access those variables in your code:

```
import os                               ## Python
int(os.environ['SLURM_NTASKS'])         ## Python

as.numeric(Sys.getenv('SLURM_NTASKS'))  ## R

str2num(getenv('SLURM_NTASKS')))        ## MATLAB
```

To use multiple cores on a node (and thereby fully utilize the node that will be exclusively assigned to your job), be careful if you only specify `--nodes`, as the environment variables will only indicate one task per node.



# Example use of standard software: Python

Let's see a basic example of doing an analysis in Python across multiple cores on multiple nodes. We'll use the airline departure data in *bayArea.csv*.

Here we'll use *IPython* for parallel computing. The example is a bit contrived in that a lot of the time is spent moving data around rather than doing computation, but it should illustrate how to do a few things.

First we'll install a Python package not already available as a module.

```
# remember to do I/O off scratch
cp bayArea.csv /global/scratch/paciorek/.
# install Python package
module load pip
# trial and error to realize which package dependencies available in modules...
module load python/2.7.8 numpy scipy six pandas pytz
pip install --user statsmodels
```

Now we'll start up an interactive session, though often this sort of thing would be done via a batch job.

```
srun -A co_stat -p savio2  --nodes=2 --ntasks-per-node=24 -t 30:0 --pty bash
```

Now we'll start up a cluster using IPython's parallel tools. To do this across multiple nodes within a SLURM job, it goes like this:
 
```
module load python/2.7.8 ipython gcc openmpi
ipcontroller --ip='*' &
sleep 5
srun ipengine &  # will start as many ipengines as we have SLURM tasks because srun is a SLURM command
sleep 15  # wait until all engines have successfully started
ipython
```

If we were doing this on a single node, we could start everything up in a single call to *ipcluster*:

```
module load python/2.7.8 ipython
ipcluster start -n $SLURM_CPUS_ON_NODE &
ipython
```

Here's our Python code (also found in *parallel.py*) for doing an analysis across multiple strata/subsets of the dataset in parallel. Note that the 'load_balanced_view' business is so that the computations are done in a load-balanced fashion, which is important for tasks that take different amounts of time to complete.

```
from IPython.parallel import Client
c = Client()
c.ids

dview = c[:]
dview.block = True
dview.apply(lambda : "Hello, World")

lview = c.load_balanced_view()
lview.block = True

import pandas
dat = pandas.read_csv('bayArea.csv', header = None)
dat.columns = ('Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','TailNum','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','Origin','Dest','Distance','TaxiIn','TaxiOut','Cancelled','CancellationCode','Diverted','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay')

dview.execute('import statsmodels.api as sm')

dat2 = dat.loc[:, ('DepDelay','Year','Dest','Origin')]
dests = dat2.Dest.unique()

mydict = dict(dat2 = dat2, dests = dests)
dview.push(mydict)

def f(id):
    sub = dat2.loc[dat2.Dest == dests[id],:]
    sub = sm.add_constant(sub)
    model = sm.OLS(sub.DepDelay, sub.loc[:,('const','Year')])
    results = model.fit()
    return results.params

import time
time.time()
parallel_result = lview.map(f, range(len(dests)))
#result = map(f, range(len(dests)))
time.time()

# some NaN values because all 'Year' values are the same for some destinations

parallel_result
```

And we'll stop our cluster. 

```
ipcluster stop
```

# Example use of standard software: Python via JupyterHub

[ HELP!! need template / instructions for use from Yong ]

# Example of hybrid parallelization with Python using threaded linear algebra

[ CP to work through this - check that Python on Savio is linked to MKL ]

# Example use of standard software: R

Let's see a basic example of doing an analysis in R across multiple cores on multiple nodes. We'll use the airline departure data in *bayArea.csv*.

We'll do this interactively though often this sort of thing would be done via a batch job.

```
# remember to do I/O off scratch
cp bayArea.csv /global/scratch/paciorek/.
module load r Rmpi
Rscript -e "install.packages('doMPI', repos = 'http://cran.cnr.berkeley.edu', lib = '/global/home/users/paciorek/R/x86_64-pc-linux-gnu-library/3.2')"

srun -A co_stat -p savio2  -N 3 --ntasks-per-node=24 -t 30:0 --pty bash
module load gcc openmpi r Rmpi
mpirun R CMD BATCH --no-save parallel-multi.R parallel-multi.Rout &
```

Now here's the R code (see *parallel-multi.R*) we're running:
```
library(doMPI)

cl = startMPIcluster()  # by default will start one fewer slave than available SLURM tasks
registerDoMPI(cl)
clusterSize(cl) # just to check

dat <- read.csv('/global/scratch/paciorek/bayArea.csv', header = FALSE,
                stringsAsFactors = FALSE)
names(dat)[16:18] <- c('delay', 'origin', 'dest')
table(dat$dest)

destVals <- unique(dat$dest)

# restrict to only columns we need to reduce copying time
dat2 <- subset(dat, select = c('delay', 'origin', 'dest'))

# some overhead in copying 'dat2' to worker processes...
results <- foreach(destVal = destVals) %dopar% {
    sub <- subset(dat2, dest == destVal)
    summary(sub$delay)
}


results

closeCluster(cl)
mpi.quit()
```

If you just want to parallelize within a node:

```
srun -A co_stat -p savio2  -N 1 -t 30:0 --pty bash
module load r
R CMD BATCH --no-save parallel-one.R parallel-one.Rout &
```

Now here's the R code (see *parallel-one.R*) we're running:
```
library(doParallel)

nCores <- as.numeric(Sys.getenv('SLURM_CPUS_ON_NODE'))
registerDoParallel(nCores)

dat <- read.csv('/global/scratch/paciorek/bayArea.csv', header = FALSE,
                stringsAsFactors = FALSE)
names(dat)[16:18] <- c('delay', 'origin', 'dest')
table(dat$dest)

destVals <- unique(dat$dest)

results <- foreach(destVal = destVals) %dopar% {
    sub <- subset(dat, dest == destVal)
    summary(sub$delay)
}

results
```

# Example of hybrid parallelization with R using threaded linear algebra

[ CP to work through this if Python+linear algebra example doesn't pan out]


# High-throughput computing
 
You may have many serial jobs to run. It may be more cost-effective to collect those jobs together and run them across multiple cores on one or more nodes.

Here are some options:

 - ht_helper (see next slide)
 - various forms of easy parallelization in Python and R 
     - some description in this document  
     - Chris Paciorek's tutorials on using [single-node parallelism](https://github.com/berkeley-scf/tutorial-parallel-basics) and [multiple-node parallelism](https://github.com/berkeley-scf/tutorial-parallel-distributed) in Python, R, and MATLAB

# ht-helper
 
More details are given in [the Savio tip on "How to run High-Throughput Computing ..."](http://research-it.berkeley.edu/services/high-performance-computing/tips-using-brc-savio-cluster)

[HELP!! - need good example and, ideally, existing code from Yong or Krishna]


# How to get additional help

 - For technical issues and questions about using Savio: 
    - brc-hpc-help@berkeley.edu
 - For questions about computing resources in general, including cloud computing: 
    - brc@berkeley.edu
 - For questions about data management (including HIPAA-protected data): 
    - researchdata@berkeley.edu


# Wrap-up

- Upcoming events (ask A Culich)

- [insert link to Savio impacts form] (ask S Masover)

- Please fill out an evaluation form



