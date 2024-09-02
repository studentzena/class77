Generate the Caption
=====================

In this activity, you will prepare caption dataset, clean it and generate caption for the image.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10598338/PCP.png" width = "480" height = "220">


Follow the given steps to complete this activity:


1. Generate the caption.


* Open the main.py file.


* Load the `best_model.h5` model.


    `model = keras.models.load_model('best_model.h5')`


* Create an empty dictonary and assign it to an varible `mapping` to store the mapped captions.


    `mapping = {}`


* Loop through every `caption`.


    `for line in tqdm(captions_doc.split('\n')):`


* Take `image_id` and `caption` from `token[0]`, `tokens[1]` respectively.


    `image_id, caption = tokens[0], tokens[1:]`


* Remove extension from image ID i.e content after `"."`, i.e split the `image_i_d` from `"."` and keep the first part.


    `image_id = image_id.split('.')[0]`


* Convert `caption` list to string using `join()`.


    `caption = " ".join(caption)`
   
* Create an empty list and assign it to a `image_id` if `image_id` key is not in `mapping`.


    `if image_id not in mapping:`


        `mapping[image_id] = []`


* Append the `caption` at `image_id` key in the mapping dictonary.


    `mapping[image_id].append(caption)`


* Read the image `113.jpg`.


    `img = cv2.imread('113.jpg')`


* Use `predict_caption()` function and pass the model, features, and tokenizer as parameters to predict the caption and store it in the variable `caption`.


    `caption = predict_caption(model, feature, tokenizer)`


* Save and run the code to check the output.


