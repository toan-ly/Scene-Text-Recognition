.PHONY: init deploy client clean streamlit

clean:
	ray stop || true
	rm -rf /tmp/ray_project/*

init: clean
	mkdir -p /tmp/ray_project
	ray start --head --include-dashboard=true --disable-usage-stats --temp-dir=/tmp/ray_project --dashboard-host=0.0.0.0

deploy:
	@echo "Starting object detection server..."
	serve run obj_detection:entrypoint --address="auto"

deploy_ocr:
	@echo "Starting OCR server..."
	serve run ocr:entrypoint --address="auto"

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app.py