class CaptioningModel(nn.Module):
	def __init__(self):
		self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

		self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(

		    "google/vit-base-patch16-224-in21k", "bert-base-uncased"

		)
		self.model.config.decoder_start_token_id = tokenizer.cls_token_id

		self.model.config.pad_token_id = tokenizer.pad_token_id

	def loss(self, image, caption):
		pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
		labels = self.tokenizer(
			text = list(caption),
            return_tensors = 'pt',
            padding = True,
            max_length = 400,
            truncation = True
			return_tensors="pt"
		).input_ids

		return self.model(pixel_values=pixel_values, labels=labels).loss

	def generate(self, image):
		pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
		output_ids = self.model.generate(pixel_values)
		preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  		preds = [pred.strip() for pred in preds]
  		return preds






