using System;
using System.Collections.Generic;
using Cloo;

// 倍精度か単精度かどっちか選択
using Real = System.Single;
//using Real = System.Double;

namespace LWisteria.StudiesOfOpenTKWithCloo.VectorAddition.UseHostPointer
{
	/// <summary>
	/// メインクラス
	/// </summary>
	static class VectorAdditionMain
	{
		/// <summary>
		/// 要素数
		/// </summary>
		const int COUNT = 1024 * 1024 * 20;

		/// <summary>
		/// 検算用ベクトル
		/// </summary>
		static Real[] answer;

		/// <summary>
		/// コマンドキュー群
		/// </summary>
		static ComputeCommandQueue[] queues;

		/// <summary>
		/// 1デバイスで計算する要素数
		/// </summary>
		static int countPerDevice = COUNT;

		#region バッファー
		/// <summary>
		/// ホストポインタ使用の計算対象1すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferHostLeft;

		/// <summary>
		/// ホストポインタ使用の計算対象2すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferHostRight;

		/// <summary>
		/// ホストポインタ使用の結果すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferHostResult;


		/// <summary>
		/// ホストポインタ不使用の計算対象1すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferNonHostLeft;

		/// <summary>
		/// ホストポインタ不使用の計算対象2すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferNonHostRight;

		/// <summary>
		/// ホストポインタ不使用の結果すべてのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferNonHostResult;


		/// <summary>
		/// ホストポインタ使用の計算対象1のバッファー群
		/// </summary>
		static ComputeSubBuffer<Real>[] buffersHostLeft;

		/// <summary>
		/// ホストポインタ使用の計算対象2のバッファー群
		/// </summary>
		static ComputeSubBuffer<Real>[] buffersHostRight;

		/// <summary>
		/// ホストポインタ使用の結果のバッファー群
		/// </summary>
		static ComputeSubBuffer<Real>[] buffersHostResult;


		/// <summary>
		/// ホストポインタ不使用の計算対象1のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersNonHostLeft;

		/// <summary>
		/// ホストポインタ不使用の計算対象2のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersNonHostRight;

		/// <summary>
		/// ホストポインタ不使用の結果のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersNonHostResult;
		#endregion

		/// <summary>
		/// 各要素を1つずつ足すカーネル
		/// </summary>
		static ComputeKernel[] addOneElement;

		/// <summary>
		/// エントリーポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("= ベクトル加算の試験 =");
			Console.WriteLine("複数GPUを使う");
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
			showResult("単一CPU                          ", false, () => SingleCpuAddition(result, left, right));
			showResult("複数CPU                          ", false, () => ParallelCpuAddition(result, left, right));
			showResult("単一GPU（ホスト）  データ転送あり", true, () => SingleGpuAdditionOneElementHost(result, left, right));
			showResult("単一GPU（ホスト）  データ転送なし", true, () => SingleGpuAdditionOneElementHost(result, left, right));
			showResult("複数GPU（ホスト）  データ転送あり", true, () => ParallelGpuAdditionOneElementHost(result, left, right));
			showResult("複数GPU（ホスト）  データ転送なし", true, () => ParallelGpuAdditionOneElementHost(result, left, right));
			showResult("単一GPU（非ホスト）データ転送あり", true, () => { WriteBuffer(left, right); SingleGpuAdditionOneElementNonHost(result, left, right); });
			showResult("単一GPU（非ホスト）データ転送なし", true, () => { SingleGpuAdditionOneElementNonHost(result, left, right); });
			showResult("複数GPU（非ホスト）データ転送あり", true, () => { WriteBuffers(left, right); ParallelGpuAdditionOneElementNonHost(result, left, right); });
			showResult("複数GPU（非ホスト）データ転送なし", true, () => ParallelGpuAdditionOneElementNonHost(result, left, right));

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

			// 1デバイスで使う要素数を計算
			countPerDevice = (int)Math.Ceiling((double)COUNT / devices.Count);

			// キューの配列を作成
			queues = new ComputeCommandQueue[devices.Count];

			// 利用可能なデバイスすべてに対して
			for(int i = 0; i < devices.Count; i++)
			{
				var device = devices[i];

				// キューを作成
				queues[i] = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);

				// デバイス情報を表示
				Console.WriteLine("* {0} ({1})", device.Name, device.Vendor);
			}

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.UseHostPointer);

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
			addOneElement = new ComputeKernel[devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				addOneElement[i] = program.CreateKernel("AddOneElement");
			}

			// バッファーを作成
			bufferHostLeft = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, left);
			bufferHostRight = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, right);
			bufferHostResult = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, result);
			bufferNonHostLeft = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, left.Length);
			bufferNonHostRight = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, right.Length);
			bufferNonHostResult = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly, result.Length);

			buffersHostLeft = new ComputeSubBuffer<Real>[devices.Count];
			buffersHostRight = new ComputeSubBuffer<Real>[devices.Count];
			buffersHostResult = new ComputeSubBuffer<Real>[devices.Count];
			buffersNonHostLeft = new ComputeBuffer<Real>[devices.Count];
			buffersNonHostRight = new ComputeBuffer<Real>[devices.Count];
			buffersNonHostResult = new ComputeBuffer<Real>[devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				buffersHostLeft[i] = new ComputeSubBuffer<Real>(bufferHostLeft, ComputeMemoryFlags.ReadOnly, countPerDevice * i, countPerDevice);
				buffersHostRight[i] = new ComputeSubBuffer<Real>(bufferHostRight, ComputeMemoryFlags.ReadOnly, countPerDevice * i, countPerDevice);
				buffersHostResult[i] = new ComputeSubBuffer<Real>(bufferHostResult, ComputeMemoryFlags.WriteOnly, countPerDevice * i, countPerDevice);

				buffersNonHostLeft[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, countPerDevice);
				buffersNonHostRight[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, countPerDevice);
				buffersNonHostResult[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly, countPerDevice);
			}
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
				// 全キューについて
				System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
				{
					// 使用するキューを設定
					var queue = queues[i];

					// 結果バッファーに初期状態を転送
					queue.WriteToBuffer(result, bufferHostResult, true, null);
					queue.WriteToBuffer(result, bufferNonHostResult, true, null);

					// データを転送
					queue.WriteToBuffer(result, buffersHostResult[i], false, countPerDevice * i, 0, countPerDevice, null);
					queue.WriteToBuffer(result, buffersNonHostResult[i], false, countPerDevice * i, 0, countPerDevice, null);
				});
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
		/// ホストポインタを使用のGPUを1つ使って1つずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionOneElementHost(Real[] result, Real[] left, Real[] right)
		{
			// 使用するキューを設定
			var queue = queues[0];

			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addOneElement[0].SetMemoryArgument(0, bufferHostResult);
			addOneElement[0].SetMemoryArgument(1, bufferHostLeft);
			addOneElement[0].SetMemoryArgument(2, bufferHostRight);

			// 計算を実行
			queue.Execute(addOneElement[0], null, new long[] { COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferHostResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}

		/// <summary>
		/// ホストポインタを不使用のGPUを1つ使って1つずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAdditionOneElementNonHost(Real[] result, Real[] left, Real[] right)
		{
			// 使用するキューを設定
			var queue = queues[0];

			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addOneElement[0].SetMemoryArgument(0, bufferNonHostResult);
			addOneElement[0].SetMemoryArgument(1, bufferNonHostLeft);
			addOneElement[0].SetMemoryArgument(2, bufferNonHostRight);

			// 計算を実行
			queue.Execute(addOneElement[0], null, new long[] { COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferNonHostResult, ref result, false, null);

			// 終了まで待機
			queue.Finish();
		}

		/// <summary>
		/// ホストポインタを使用のGPUを複数使って1つずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void ParallelGpuAdditionOneElementHost(Real[] result, Real[] left, Real[] right)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 引数を設定
				//  # 結果を格納するベクトル
				//  # 計算対象のベクトル1
				//  # 計算対象のベクトル2
				addOneElement[i].SetMemoryArgument(0, buffersHostResult[i]);
				addOneElement[i].SetMemoryArgument(1, buffersHostLeft[i]);
				addOneElement[i].SetMemoryArgument(2, buffersHostRight[i]);

				// 計算を実行
				queues[i].Execute(addOneElement[i], null, new long[] { countPerDevice }, null, null);

				// 結果を読み込み
				queues[i].ReadFromBuffer(buffersHostResult[i], ref result, false, 0, countPerDevice * i, countPerDevice, null);

				// 終了まで待機
				queues[i].Finish();
			});
		}

		/// <summary>
		/// ホストポインタを不使用のGPUを複数使って1つずつ加算を実行する
		/// </summary>
		/// <param name="result">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void ParallelGpuAdditionOneElementNonHost(Real[] result, Real[] left, Real[] right)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 引数を設定
				//  # 結果を格納するベクトル
				//  # 計算対象のベクトル1
				//  # 計算対象のベクトル2
				addOneElement[i].SetMemoryArgument(0, buffersNonHostResult[i]);
				addOneElement[i].SetMemoryArgument(1, buffersNonHostLeft[i]);
				addOneElement[i].SetMemoryArgument(2, buffersNonHostRight[i]);

				// 計算を実行
				queues[i].Execute(addOneElement[i], null, new long[] { countPerDevice }, null, null);

				// 結果を読み込み
				queues[i].ReadFromBuffer(buffersNonHostResult[i], ref result, false, 0, countPerDevice * i, countPerDevice, null);

				// 終了まで待機
				queues[i].Finish();
			});
		}

		/// <summary>
		/// 単一GPUに全データを転送する
		/// </summary>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void WriteBuffer(Real[] left, Real[] right)
		{
			// 使用するキューを設定
			var queue = queues[0];

			// データを転送
			queue.WriteToBuffer(left, bufferNonHostLeft, false, null);
			queue.WriteToBuffer(right, bufferNonHostRight, false, null);
		}

		/// <summary>
		/// 複数GPUに各データを転送する
		/// </summary>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void WriteBuffers(Real[] left, Real[] right)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 使用するキューを設定
				var queue = queues[i];

				// データを転送
				queue.WriteToBuffer(left, buffersNonHostLeft[i], false, countPerDevice * i, 0, countPerDevice, null);
				queue.WriteToBuffer(right, buffersNonHostRight[i], false, countPerDevice * i, 0, countPerDevice, null);
			});
		}
	}
}