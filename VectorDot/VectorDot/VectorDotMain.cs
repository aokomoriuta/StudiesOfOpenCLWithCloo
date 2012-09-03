using System;
using System.Collections.Generic;
using Cloo;

// 倍精度か単精度かどっちか選択
//using Real = System.Single;
using Real = System.Double;

namespace LWisteria.StudiesOfOpenTKWithCloo.VectorDot
{
	/// <summary>
	/// メインクラス
	/// </summary>
	static class VectorAdditionMain
	{
		/// <summary>
		/// 要素数
		/// </summary>
		const int COUNT = 21 * 1024 * 1024;

		/// <summary>
		/// 検算値
		/// </summary>
		static Real answer;

		/// <summary>
		/// コマンドキュー群
		/// </summary>
		static ComputeCommandQueue[] queues;

		/// <summary>
		/// 1デバイスで計算する要素数
		/// </summary>
		static int countPerDevice = COUNT;

		/// <summary>
		/// ワークグループ内ワークアイテム数
		/// </summary>
		static int localSize = 1;

		/// <summary>
		/// CPUで並列させる数
		/// </summary>
		static int taskCount = 1;

		/// <summary>
		/// 1タスク内での結果
		/// </summary>
		static Real[] resultsPerTask;

		/// <summary>
		/// 1デバイス内での結果
		/// </summary>
		static Real[] resultsPerDevice;

		#region バッファー
		/// <summary>
		/// 計算対象1のバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferLeft;

		/// <summary>
		/// 計算対象2のバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferRight;

		/// <summary>
		/// 結果のバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferResult;


		/// <summary>
		/// 計算対象1のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersLeft;

		/// <summary>
		/// 計算対象2のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersRight;

		/// <summary>
		/// 結果のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersResult;
		#endregion

		/// <summary>
		/// 各要素同士の積を計算するカーネル
		/// </summary>
		static ComputeKernel[] multyplyEachElement;

		/// <summary>
		/// 全要素の和を計算するカーネル
		/// </summary>
		/// <remarks>
		/// ver 0: グローバルメモリのまま使用
		/// ver 1: ローカルメモリ使用
		/// ver 2: 前半部分を後半部分に足す
		/// ver 3: ローカルへの複製のみのワークアイテムをなくして、数を減らす
		/// ver 4: 2倍および1/2倍をビットシフト演算にする
		/// </remarks>
		static ComputeKernel[,] reductionSum;

		/// <summary>
		/// リダクションを実装しているバージョン数
		/// </summary>
		const int REDUCTION_VERSION = 4;

		/// <summary>
		/// エントリーポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("= ベクトル内積の試験 =");
			Console.WriteLine();

			// 計算対象および結果を格納するベクトルを作成
			var left = new Real[COUNT];
			var right = new Real[COUNT];

			// 検算値を初期化
			answer = 0;

			// 計算対象のデータを作成
			for(int i = 0; i < COUNT; i++)
			{
				left[i] = (Real)i / 10;// Math.Cos(i);
				right[i] = (Real)1;// Math.Sin(i);

				answer += left[i] * right[i];
			}

			// CPUの
			foreach(var processor in new System.Management.ManagementClass("Win32_Processor").GetInstances())
			{
				// コア数でタスク数を設定
				taskCount = int.Parse(processor["NumberOfCores"].ToString());

				// データを表示
				Console.WriteLine("CPU：{0} ({1}コア)", processor["Name"], taskCount);
			}

			// 各タスクでの結果を初期化
			resultsPerTask = new Real[taskCount];

			// OpenCLの使用準備
			InitializeOpenCL(left, right);

			// 実行開始
			Console.WriteLine();
			Console.WriteLine("== 実行速度計測開始 ==");
			Console.WriteLine("要素数：{0} ({1}[MB])", COUNT, COUNT * sizeof(Real) / 1024 / 1024);
			Action<string, bool, Func<Real>> showResult = (name, useGpu, action) =>
				Console.WriteLine("{0}: {1,12}", name, ProcessingTime(action, useGpu));

			// 各方法で実行して結果を表示
			showResult("単一CPU              ", false, () => VectorDotSingleCpu(left, right));
			showResult("複数CPU              ", false, () => VectorDotParallelCpu(left, right));
			showResult("単一GPU（計算 ver.0）", true, () => VectorDotSingleGpu0());
			showResult("単一GPU（計算 ver.1）", true, () => VectorDotSingleGpu1());
			showResult("単一GPU（計算 ver.2）", true, () => VectorDotSingleGpu2());
			showResult("単一GPU（計算 ver.3）", true, () => VectorDotSingleGpu3());
			showResult("単一GPU（計算 ver.4）", true, () => VectorDotSingleGpu4());

			// もう1回
			Console.WriteLine("---");
			showResult("単一GPU（計算 ver.0）", true, () => VectorDotSingleGpu0());
			showResult("単一GPU（計算 ver.1）", true, () => VectorDotSingleGpu1());
			showResult("単一GPU（計算 ver.2）", true, () => VectorDotSingleGpu2());
			showResult("単一GPU（計算 ver.3）", true, () => VectorDotSingleGpu3());
			showResult("単一GPU（計算 ver.4）", true, () => VectorDotSingleGpu4());

			// 成功で終了
			return System.Environment.ExitCode;
		}

		/// <summary>
		/// OpenCL関係の準備をする
		/// </summary>
		static void InitializeOpenCL(Real[] left, Real[] right)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];
			Console.WriteLine("プラットフォーム：{0} ({1})", platform.Name, platform.Version);

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			Console.WriteLine("デバイス数：{0}", devices.Count);

			// 1デバイスで使う要素数を計算
			countPerDevice = (int)Math.Ceiling((double)COUNT / devices.Count);

			// デバイス内での結果を作成
			resultsPerDevice = new Real[devices.Count];

			// ワークグループ内ワークアイテム数
			localSize = (int)devices[0].MaxWorkItemSizes[0];

			// キューの配列を作成
			queues = new ComputeCommandQueue[devices.Count];

			// 利用可能なデバイスすべてに対して
			for(int i = 0; i < devices.Count; i++)
			{
				// デバイスを取得
				var device = devices[i];

				// キューを作成
				queues[i] = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);

				// デバイス情報を表示
				Console.WriteLine("* {0} ({1})", device.Name, device.Vendor);
			}

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.VectorDot);

			// ビルドしてみて
			try
			{
				string realString = ((typeof(Real) == typeof(Double)) ? "double" : "float");

				program.Build(devices,
					string.Format(" -D REAL={0} -Werror", realString),
					null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// ログを表示して例外を投げる
				throw new ApplicationException(string.Format("{0}\n{1}", ex.Message, program.GetBuildLog(devices[0])), ex);
			}

			// カーネルを作成
			multyplyEachElement = new ComputeKernel[devices.Count];
			reductionSum = new ComputeKernel[REDUCTION_VERSION+1, devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				multyplyEachElement[i] = program.CreateKernel("MultyplyEachElement");

				reductionSum[0, i] = program.CreateKernel("ReductionSum0");
				reductionSum[1, i] = program.CreateKernel("ReductionSum1");
				reductionSum[2, i] = program.CreateKernel("ReductionSum2");
				reductionSum[3, i] = program.CreateKernel("ReductionSum3");
				reductionSum[4, i] = program.CreateKernel("ReductionSum4");
			}

			// 単一GPU用バッファーを作成
			bufferLeft = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, left);
			bufferRight = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, right);
			bufferResult = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadWrite, left.Length);

			// 複数GPU用バッファーを作成
			buffersLeft = new ComputeBuffer<Real>[devices.Count];
			buffersRight = new ComputeBuffer<Real>[devices.Count];
			buffersResult = new ComputeBuffer<Real>[devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				buffersLeft[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, countPerDevice);
				buffersRight[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, countPerDevice);
				buffersResult[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly, countPerDevice);
			}
		}

		/// <summary>
		/// 操作の実行時間を計測する
		/// </summary>
		/// <param name="action">計測対象の操作</param>
		/// <returns>実行時間（ミリ秒）</returns>
		static long ProcessingTime(Func<Real> action, bool useGpu)
		{
			// ストップウォッチ作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			// GPUなら
			if(useGpu)
			{
				//// 全キューについて
				//System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
				//{
				//    // 使用するキューを設定
				//    var queue = queues[i];

				//    // 結果バッファーに初期状態を転送
				//    //queue.WriteToBuffer(result, buffersNonHostResult[i], false, 0, 0, 1, null);
				//});
			}
			


			// 計測開始
			stopwatch.Restart();

			// 操作を実行
			Real result = action();

			// 計測終了
			stopwatch.Stop();


			// 精度以上の誤差があれば
			if(result - answer > Math.Pow(10, (Math.Log10(answer) - 8)))
			{
				// 誤差を通知出力
				Console.WriteLine("result={0:e} vs answer={1:e} ({2:e})", result, answer, result - answer);
			}

			// 実行時間を返す
			return stopwatch.ElapsedTicks;
		}


		/// <summary>
		/// CPUで並列させずに内積を計算する
		/// </summary>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleCpu(Real[] left, Real[] right)
		{
			Real result = 0;

			// 全要素を順に
			for(int i = 0; i < COUNT; i++)
			{
				// 足す
				result += left[i] * right[i];
			}

			return result;
		}

		/// <summary>
		/// CPUで並列させて内積を計算する
		/// </summary>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotParallelCpu(Real[] left, Real[] right)
		{
			Real result = 0;

			// 1タスクに割り当てる数を計算
			int countPerTask = (int)System.Math.Ceiling((double)COUNT / taskCount);

			// 全タスクで
			System.Threading.Tasks.Parallel.For(0, taskCount, (n) =>
			{
				// 全要素を順に
				for(int i = n * countPerTask; i < System.Math.Min(COUNT, (n + 1) * countPerTask); i++)
				{
					// 足す
					resultsPerTask[n] += left[i] * right[i];
				}
			});

			// 全タスクの結果を
			foreach(var resultOfTask in resultsPerTask)
			{
				// 結果に加える
				result += resultOfTask;
			}

			return result;
		}


		/// <summary>
		///	GPUを1つ使って内積を計算する（リダクションのバージョン0使用）
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleGpu0()
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			multyplyEachElement[0].SetMemoryArgument(0, bufferResult);
			multyplyEachElement[0].SetMemoryArgument(1, bufferLeft);
			multyplyEachElement[0].SetMemoryArgument(2, bufferRight);
			queues[0].Execute(multyplyEachElement[0], null, new long[] { COUNT }, null, null);

			// リダクションを繰り返す
			for(int n = 1; n < COUNT; n *= 2)
			{
				// ワークアイテム数を計算
				long globalSize = Math.Max((long)Math.Ceiling(COUNT / 2.0 / n), 1);

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 要素数
				//  # 隣までの距離
				reductionSum[0,0].SetMemoryArgument(0, bufferResult);
				reductionSum[0,0].SetValueArgument(1, COUNT);
				reductionSum[0,0].SetValueArgument(2, n);
				queues[0].Execute(reductionSum[0,0], null, new long[] { globalSize }, null, null);
			}

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref resultsPerDevice, false, 0, 0, 1, null);

			// 終了まで待機
			queues[0].Finish();

			return resultsPerDevice[0];
		}

		/// <summary>
		///	GPUを1つ使って内積を計算する（リダクションのバージョン1使用）
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleGpu1()
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			multyplyEachElement[0].SetMemoryArgument(0, bufferResult);
			multyplyEachElement[0].SetMemoryArgument(1, bufferLeft);
			multyplyEachElement[0].SetMemoryArgument(2, bufferRight);
			queues[0].Execute(multyplyEachElement[0], null, new long[] { COUNT }, null, null);

			// 計算する配列の要素数
			int targetSize = COUNT;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum[1, 0].SetMemoryArgument(0, bufferResult);
				reductionSum[1, 0].SetValueArgument(1, targetSize);
				reductionSum[1, 0].SetLocalArgument(2, sizeof(Real) * localSize);
				queues[0].Execute(reductionSum[1, 0], null, new long[] { globalSize }, new long[] { localSize }, null);

				// 次の配列の要素数を今のワークアイテム数にする
				 targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref resultsPerDevice, false, 0, 0, 1, null);

			// 終了まで待機
			queues[0].Finish();

			return resultsPerDevice[0];
		}
		
		/// <summary>
		///	GPUを1つ使って内積を計算する（リダクションのバージョン2使用）
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleGpu2()
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			multyplyEachElement[0].SetMemoryArgument(0, bufferResult);
			multyplyEachElement[0].SetMemoryArgument(1, bufferLeft);
			multyplyEachElement[0].SetMemoryArgument(2, bufferRight);
			queues[0].Execute(multyplyEachElement[0], null, new long[] { COUNT }, null, null);

			// 計算する配列の要素数
			int targetSize = COUNT;
			
			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum[2, 0].SetMemoryArgument(0, bufferResult);
				reductionSum[2, 0].SetValueArgument(1, targetSize);
				reductionSum[2, 0].SetLocalArgument(2, sizeof(Real) * localSize);
				queues[0].Execute(reductionSum[2, 0], null, new long[] { globalSize }, new long[] { localSize }, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref resultsPerDevice, false, 0, 0, 1, null);

			// 終了まで待機
			queues[0].Finish();

			return resultsPerDevice[0];
		}

		/// <summary>
		///	GPUを1つ使って内積を計算する（リダクションのバージョン3使用）
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleGpu3()
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			multyplyEachElement[0].SetMemoryArgument(0, bufferResult);
			multyplyEachElement[0].SetMemoryArgument(1, bufferLeft);
			multyplyEachElement[0].SetMemoryArgument(2, bufferRight);
			queues[0].Execute(multyplyEachElement[0], null, new long[] { COUNT }, null, null);

			// 計算する配列の要素数
			int targetSize = COUNT;

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum[3, 0].SetMemoryArgument(0, bufferResult);
				reductionSum[3, 0].SetValueArgument(1, targetSize);
				reductionSum[3, 0].SetLocalArgument(2, sizeof(Real) * localSize);
				queues[0].Execute(reductionSum[3, 0], null, new long[] { globalSize }, new long[] { localSize }, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref resultsPerDevice, false, 0, 0, 1, null);

			// 終了まで待機
			queues[0].Finish();

			return resultsPerDevice[0];
		}

		/// <summary>
		///	GPUを1つ使って内積を計算する（リダクションのバージョン4使用）
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static Real VectorDotSingleGpu4()
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			multyplyEachElement[0].SetMemoryArgument(0, bufferResult);
			multyplyEachElement[0].SetMemoryArgument(1, bufferLeft);
			multyplyEachElement[0].SetMemoryArgument(2, bufferRight);
			queues[0].Execute(multyplyEachElement[0], null, new long[] { COUNT }, null, null);

			// 計算する配列の要素数
			int targetSize = COUNT;

			//var debug = new Real[COUNT];
			//queues[0].ReadFromBuffer(bufferResult, ref debug, true, null);

			// 計算する配列の要素数が1以上の間
			while(targetSize > 1)
			{
				// ワークアイテム数を計算
				int globalSize = (int)Math.Ceiling((double)targetSize / 2 / localSize) * localSize;

				// 隣との和を計算
				//  # 和を計算するベクトル
				//  # 計算する要素数
				//  # ローカルメモリ
				reductionSum[4, 0].SetMemoryArgument(0, bufferResult);
				reductionSum[4, 0].SetValueArgument(1, targetSize);
				reductionSum[4, 0].SetLocalArgument(2, sizeof(Real) * localSize);
				queues[0].Execute(reductionSum[4, 0], null, new long[] { globalSize }, new long[] { localSize }, null);


				//queues[0].ReadFromBuffer(bufferResult, ref debug, true, 0, 0, targetSize, null);

				// 次の配列の要素数を今のワークアイテム数にする
				targetSize = globalSize / localSize;
			}

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref resultsPerDevice, false, 0, 0, 1, null);

			// 終了まで待機
			queues[0].Finish();

			return resultsPerDevice[0];
		}
	}
}