﻿using System;
using Cloo;

// 倍精度か単精度かどっちか選択
using Real = System.Single;
//using Real = System.Double;

namespace LWisteria.StudiesOfOpenTKWithCloo.VectorAddition.HeavyWorkItem
{
	/// <summary>
	/// メインクラス
	/// </summary>
	static class HeavyWorkItemMain
	{
		/// <summary>
		/// ベクトルとして扱う場合の要素数
		/// </summary>
		const int VECTOR_COUNT = 16;

		/// <summary>
		/// 1ワークアイテムが計算する要素数
		/// </summary>
		const int COUNT_PER_WORKITEM = 16;

		/// <summary>
		/// 要素数
		/// </summary>
		const int COUNT = 1024*100 * VECTOR_COUNT * COUNT_PER_WORKITEM;

		/// <summary>
		/// 検算用ベクトル
		/// </summary>
		static Real[] answer;

		/// <summary>
		/// コマンドキュー群
		/// </summary>
		static ComputeCommandQueue queue;

		#region バッファー
		/// <summary>
		/// 計算対象1すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferLeft;

		/// <summary>
		/// 計算対象2すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferRight;

		/// <summary>
		/// 結果すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferResult;
		#endregion

		#region カーネル
		/// <summary>
		/// 各要素を1つずつ足すカーネル
		/// </summary>
		static ComputeKernel addOneElement;

		/// <summary>
		/// ベクトルとして1つずつ足すカーネル
		/// </summary>
		static ComputeKernel addOneVector;

		/// <summary>
		/// 各要素を複数ずつ足すカーネル
		/// </summary>
		static ComputeKernel addMoreElement;

		/// <summary>
		/// ベクトルとして複数ずつ足すカーネル
		/// </summary>
		static ComputeKernel addMoreVector;
		#endregion

		/// <summary>
		/// エントリーポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("= ベクトル加算の試験 =");
			Console.WriteLine("1ワークアイテムの処理量を変えてみる");
			Console.WriteLine();

			// 計算対象および結果を格納するベクトルを作成
			var left = new Real[COUNT];
			var right = new Real[COUNT];
			var result = new Real[COUNT];

			// 検算用ベクトル
			answer = new Real[COUNT];

			// 計算対象のデータを作成
			for(int i = 0; i < COUNT; i++)
			{
				left[i] = (Real)i / 10000;
				right[i] = (Real)i * 0 / COUNT;

				answer[i] = left[i] + right[i];
			}

			// OpenCLの使用準備
			InitializeOpenCL(result, left, right);

			// 実行開始
			Console.WriteLine();
			Console.WriteLine("== 実行速度計測開始 ==");
			Console.WriteLine("要素数：{0} ({1}[MB])", COUNT, COUNT * sizeof(Real) / 1024 / 1024);
			Action<string, bool, Action> showResult = (name, useGpu, action) =>
				Console.WriteLine("{0}: {1,12}", name, ProcessingTime(action, useGpu, result));

			// 各方法で実行して結果を表示
			showResult("単一CPU                ", false, () => SingleCpuAddition(result, left, right));
			showResult("複数CPU                ", false, () => ParallelCpuAddition(result, left, right));
			showResult("単一GPU（各要素）      ", true, () => SingleGpuAdditionOneElement(result, left, right));
			showResult("単一GPU（ベクトル）    ", true, () => SingleGpuAdditionOneVector(result, left, right));
			showResult("単一GPU（複数要素）    ", true, () => SingleGpuAdditionMoreElement(result, left, right));
			showResult("単一GPU（複数ベクトル）", true, () => SingleGpuAdditionMoreVector(result, left, right));

			// もう1回
			Console.WriteLine("----");
			showResult("単一CPU                ", false, () => SingleCpuAddition(result, left, right));
			showResult("複数CPU                ", false, () => ParallelCpuAddition(result, left, right));
			showResult("単一GPU（各要素）      ", true, () => SingleGpuAdditionOneElement(result, left, right));
			showResult("単一GPU（ベクトル）    ", true, () => SingleGpuAdditionOneVector(result, left, right));
			showResult("単一GPU（複数要素）    ", true, () => SingleGpuAdditionMoreElement(result, left, right));
			showResult("単一GPU（複数ベクトル）", true, () => SingleGpuAdditionMoreVector(result, left, right));

			// 成功で終了
			return System.Environment.ExitCode;
		}

		/// <summary>
		/// OpenCL関係の準備をする
		/// </summary>
		static void InitializeOpenCL(Real[] result, Real[] left, Real[] right)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];
			Console.WriteLine("プラットフォーム：{0} ({1})", platform.Name, platform.Version);

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			Console.WriteLine("デバイス数：{0}", devices.Count);

			// キューを作成
			queue = new ComputeCommandQueue(context, devices[0], ComputeCommandQueueFlags.None);

			// 利用可能なデバイスすべてに対して
			for(int i = 0; i < devices.Count; i++)
			{
				var device = devices[i];

				// デバイス情報を表示
				Console.WriteLine("* {0} ({1})", device.Name, device.Vendor);
			}

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.HeavyWorkItem);

			// ビルドしてみて
			try
			{
				string realString = ((typeof(Real) == typeof(Double)) ? "double" : "float");

				program.Build(devices,
					string.Format(" -D REAL={0} -D REALV={0}{1} -D VLOADN=vload{1} -D VSTOREN=vstore{1} -D COUNT_PER_WORKITEM={2} -Werror", realString, VECTOR_COUNT, COUNT_PER_WORKITEM),
					null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// ログを表示して例外を投げる
				throw new ApplicationException(string.Format("{0}\n{1}", ex.Message, program.GetBuildLog(devices[0])), ex);
			}

			// カーネルを作成
			addOneElement =  program.CreateKernel("AddOneElement");
			addOneVector = program.CreateKernel("AddOneVector");
			addMoreElement = program.CreateKernel("AddMoreElement");
			addMoreVector = program.CreateKernel("AddMoreVector");

			// バッファーを作成
			bufferLeft = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, left);
			bufferRight = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, right);
			bufferResult = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, result);
		}

		/// <summary>
		/// 操作の実行時間を計測する
		/// </summary>
		/// <param name="action">計測対象の操作</param>
		/// <returns>実行時間（ミリ秒）</returns>
		static long ProcessingTime(Action action, bool useGpu, Real[] result)
		{
			// ストップウォッチ作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			// すべての結果を
			for(int i = 0; i < COUNT; i++)
			{
				// 0に初期化
				result[i] = 0;
			}

			// GPUなら
			if(useGpu)
			{
				// 結果バッファーに初期状態を転送
				queue.WriteToBuffer(result, bufferResult, true, null);
			}


			// 計測開始
			stopwatch.Restart();

			// 操作を実行
			action();

			// 計測終了
			stopwatch.Stop();


			// すべての結果を
			for(int i = 0; i < COUNT; i++)
			{
				// 検算して違えば
				if(result[i] != answer[i])
				{
					// 出力
					Console.WriteLine("{0,8}: r={1,5:f} vs a={2,5:f}", i, result[i], answer[i]); 
				}
			}

			// 実行時間を返す
			return stopwatch.ElapsedTicks;
		}


		/// <summary>
		/// CPUで並列させずに加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleCpuAddition(Real[] result, Real[] left, Real[] right)
		{
			// 全要素を順に
			for(int i = 0; i < COUNT; i++)
			{
				// 足す
				result[i] = left[i] + right[i];
			}
		}

		/// <summary>
		/// CPUで並列させて加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void ParallelCpuAddition(Real[] result, Real[] left, Real[] right)
		{
			// 全要素を並列で
			System.Threading.Tasks.Parallel.For(0, COUNT, (i) =>
			{
				// 足す
				result[i] = left[i] + right[i];
			});
		}

		/// <summary>
		/// GPUを1つ使って1つずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionOneElement(Real[] result, Real[] left, Real[] right)
		{
			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addOneElement.SetMemoryArgument(0, bufferResult);
			addOneElement.SetMemoryArgument(1, bufferLeft);
			addOneElement.SetMemoryArgument(2, bufferRight);

			// 計算を実行
			queue.Execute(addOneElement, null, new long[] { COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}

		/// <summary>
		/// GPUを1つ使ってベクトルとして加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionOneVector(Real[] result, Real[] left, Real[] right)
		{
			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addOneVector.SetMemoryArgument(0, bufferResult);
			addOneVector.SetMemoryArgument(1, bufferLeft);
			addOneVector.SetMemoryArgument(2, bufferRight);

			// 計算を実行
			queue.Execute(addOneVector, null, new long[] { COUNT / VECTOR_COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}

		/// <summary>
		/// GPUを1つ使って複数ずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionMoreElement(Real[] result, Real[] left, Real[] right)
		{
			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addMoreElement.SetMemoryArgument(0, bufferResult);
			addMoreElement.SetMemoryArgument(1, bufferLeft);
			addMoreElement.SetMemoryArgument(2, bufferRight);

			// 計算を実行
			queue.Execute(addMoreElement, null, new long[] { COUNT/COUNT_PER_WORKITEM }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}

		/// <summary>
		/// GPUを1つ使って複数ずつベクトルとして加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionMoreVector(Real[] result, Real[] left, Real[] right)
		{
			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addMoreVector.SetMemoryArgument(0, bufferResult);
			addMoreVector.SetMemoryArgument(1, bufferLeft);
			addMoreVector.SetMemoryArgument(2, bufferRight);

			// 計算を実行
			queue.Execute(addMoreVector, null, new long[] { COUNT / COUNT_PER_WORKITEM / VECTOR_COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}
	}
}