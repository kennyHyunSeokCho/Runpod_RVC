import os
import sys
import argparse
import subprocess

# 한글 주석: v2_core 원클릭 파이썬 실행 스크립트
# - URL 모드: auto_train.py 사용 (다운로드→전처리→특징추출→학습)
# - 로컬 모드: core.py의 subcommand를 순차 실행

def run(cmd: list[str]):
	"""한글 주석: 명령 실행 유틸리티(에러 발생 시 즉시 종료)"""
	print("$ ", " ".join(cmd))
	ret = subprocess.run(cmd)
	if ret.returncode != 0:
		sys.exit(ret.returncode)


def main():
	parser = argparse.ArgumentParser(description="v2_core one-shot runner")
	parser.add_argument("--model_name", required=True, type=str)
	parser.add_argument("--dataset_path", type=str, default="")
	parser.add_argument("--dataset_url", type=str, default="")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--gpu", type=str, default="0")
	parser.add_argument("--cpu_cores", type=int, default=os.cpu_count() or 4)
	parser.add_argument("--index_algorithm", type=str, default="Auto")
	args = parser.parse_args()

	python = sys.executable
	root = os.getcwd()
	core_py = os.path.join(root, "rvc", "v2_core", "core.py")
	auto_py = os.path.join(root, "rvc", "v2_core", "auto_train.py")

	# URL 모드: auto_train 사용
	if args.dataset_url:
		print(f"==== AUTO: {args.model_name} (URL) ====")
		run([
			python,
			auto_py,
			"--model_name", args.model_name,
			"--dataset_url", args.dataset_url,
			"--batch_size", str(args.batch_size),
			"--gpu", args.gpu,
			"--index_algorithm", args.index_algorithm,
		])
		print("==== DONE ====")
		print(f"- Logs:   {os.path.join(root, 'logs', args.model_name)}")
		print(f"- Index:  {os.path.join(root, 'logs', args.model_name, args.model_name + '.index')}")
		return

	# 로컬 모드: dataset_path 미지정 시 기본값
	dataset_path = args.dataset_path or os.path.join("assets", "datasets", args.model_name)
	print(f"==== PIPELINE: {args.model_name} (LOCAL) ====")

	# 전처리
	run([
		python, core_py, "preprocess",
		"--model_name", args.model_name,
		"--dataset_path", dataset_path,
		"--sample_rate", "48000",
		"--cpu_cores", str(args.cpu_cores),
	])

	# 특징 추출
	run([
		python, core_py, "extract",
		"--model_name", args.model_name,
		"--f0_method", "rmvpe",
		"--cpu_cores", str(args.cpu_cores),
		"--gpu", args.gpu,
		"--sample_rate", "48000",
		"--embedder_model", "contentvec",
		"--include_mutes", "2",
	])

	# 학습 (48k/epoch500/중간세이브 없음)
	run([
		python, core_py, "train",
		"--model_name", args.model_name,
		"--batch_size", str(args.batch_size),
		"--gpu", args.gpu,
		"--index_algorithm", args.index_algorithm,
	])

	print("==== DONE ====")
	print(f"- Logs:   {os.path.join(root, 'logs', args.model_name)}")
	print(f"- Index:  {os.path.join(root, 'logs', args.model_name, args.model_name + '.index')}")


if __name__ == "__main__":
	main()
