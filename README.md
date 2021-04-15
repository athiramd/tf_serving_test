# Experiments

#Train and serve a Fashion model with TensorFlow Serving 

* Create virtual enviornment

```virtualenv venv --python=python3```


* Activate virtualenviornment

```source venv/bin/activate```

* Install Requirements

```pip install -r requirements.txt```

* Run:

```python tf_serving.py```

* To test:

	* Open another terminal

		* Activate enviornment

		* nohup tensorflow_model_server   --rest_api_port=8501   --model_name=fashion_model   --model_base_path="full path to pb folder" >server.log 2>&1

		* Check server log status is success

	* If Server log status is success
 
 	```python tf_serving.py``` -in predict mode in another terminal



