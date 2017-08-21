FROM jupyter/datascience-notebook:latest

ENV JOBLIB_TEMP_FOLDER /cognoma/machine-learning/job-lib
EXPOSE 8888

RUN conda install vega --channel conda-forge
RUN conda install dask-searchcv --channel conda-forge
# RUN pip install git+git://github.com/altair-viz/jupyter_vega.git
# RUN jupyter nbextension install --py --sys-prefix jupyter_vega
# RUN jupyter nbextension enable --py --sys-prefix jupyter_vega

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''"]
