FROM spark-py
RUN pip3 install holidays
RUN pip3 install lunarcalendar
RUN pip3 install convertdate
RUN pip3 install pandas
RUN pip3 show pandas
RUN pip3 list -v
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install pystan
RUN pip3 install fbprophet
RUN pip3 install pyarrow==0.14.1
RUN pip3 install fsspec
RUN pip3 install s3fs
RUN mkdir -p /opt/jobs/
COPY sparkts.py /opt/jobs/
