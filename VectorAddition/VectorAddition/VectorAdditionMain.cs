using System;
using System.Collections.Generic;
using Cloo;

// 倍精度か単精度かどっちか選択
//using Real = System.Single;
using Real = System.Double;

namespace LWisteria.StudiesOfOpenTKWithCloo.VectorAddition
{
	/// <summary>
	/// メインクラス
	/// </summary>
	static class VectorAdditionMain
	{
		/// <summary>
		/// 要素数
		/// </summary>
		const int COUNT = 20000000;

		/// <summary>
		/// コマンドキュー群
		/// </summary>
		static List<ComputeCommandQueue> queues;

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
		static ComputeBuffer<Real> bufferAnswer;
		#endregion

		#region カーネル
		/// <summary>
		/// 各要素をそれぞれ足すカーネル
		/// </summary>
		static ComputeKernel addEachVector;
		#endregion

		/// <summary>
		/// エントリーポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("= ベクトル加算の試験 =");

			// 計算対象および結果を格納するベクトルを作成
			var left = new Real[COUNT];
			var right = new Real[COUNT];
			var answer = new Real[COUNT];

			// 計算対象のデータを作成
			for(int i = 0; i < COUNT; i++)
			{
				left[i] = i / COUNT;
				right[i] = i * i / COUNT;
			}

			// OpenCLの使用準備
			InitializeOpenCL();

			// 実行開始
			Console.WriteLine("== 実行速度計測開始 ==");
			Console.WriteLine("要素数：{0}", COUNT);

			// 各方法で実行して結果を表示
			Console.WriteLine("単一CPU: {0,8}[ms]", ProcessingTime(() => SingleCpuAddition(answer, left, right)));
			Console.WriteLine("複数CPU: {0,8}[ms]", ProcessingTime(() => ParallelCpuAddition(answer, left, right)));
			Console.WriteLine("単一GPU: {0,8}[ms]", ProcessingTime(() => SingleGpuAddition(answer, left, right)));

			// 成功で終了
			return System.Environment.ExitCode;
		}

		/// <summary>
		/// OpenCL関係の準備をする
		/// </summary>
		static void InitializeOpenCL()
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];
			Console.WriteLine("プラットフォーム：{0} ({1})", platform.Name, platform.Version);

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			Console.WriteLine("デバイス数：{0}", devices.Count);

			// キューの配列を作成
			queues = new List<ComputeCommandQueue>(devices.Count);

			// 利用可能なデバイスすべてに対して
			foreach(var device in devices)
			{
				// キューを作成
				queues.Add(new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None));

				// デバイス情報を表示
				Console.WriteLine("* {0} ({1})", device.Name, device.Vendor);
			}

			// プログラムを作成
			var program = new ComputeProgram(context, Properties.Resources.VectorAddition);

			// ビルドしてみて
			try
			{
				string realString = ((typeof(Real) == typeof(Double)) ? "double" : "float");

				program.Build(devices,
					"-D REAL=" + realString + " -Werror", null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// ログを表示して例外を投げる
				throw new ApplicationException(string.Format("{0}\n{1}", ex.Message, program.GetBuildLog(devices[0])), ex);
			}

			// カーネルを作成
			addEachVector = program.CreateKernel("AddEachVector");

			// バッファーを作成
			bufferLeft = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, COUNT);
			bufferRight = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, COUNT);
			bufferAnswer = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly, COUNT);
		}

		/// <summary>
		/// 操作の実行時間を計測する
		/// </summary>
		/// <param name="action">計測対象の操作</param>
		/// <returns>実行時間（ミリ秒）</returns>
		static long ProcessingTime(Action action)
		{
			// ストップウォッチ作成
			var stopwatch = new System.Diagnostics.Stopwatch();

			// 計測開始
			stopwatch.Restart();

			// 操作を実行
			action();

			// 実行時間を返す
			return stopwatch.ElapsedMilliseconds;
		}

		/// <summary>
		/// CPUで並列させずに加算を実行する
		/// </summary>
		/// <param name="answer">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleCpuAddition(Real[] answer, Real[] left, Real[] right)
		{
			// 全要素を順に
			for(int i = 0; i < COUNT; i++)
			{
				// 足す
				answer[i] = left[i] + right[i];
			}
		}

		/// <summary>
		/// CPUで並列させて加算を実行する
		/// </summary>
		/// <param name="answer">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void ParallelCpuAddition(Real[] answer, Real[] left, Real[] right)
		{
			// 全要素を並列で
			System.Threading.Tasks.Parallel.For(0, COUNT, (i) =>
			{
				// 足す
				answer[i] = left[i] + right[i];
			});
		}

		/// <summary>
		/// GPUを1つ使って加算を実行する
		/// </summary>
		/// <param name="answer">結果を格納する対象</param>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void SingleGpuAddition(Real[] answer, Real[] left, Real[] right)
		{
			// 使用するキューを設定
			var queue = queues[0];

			// 計算対象のデータを転送
			queue.WriteToBuffer(left,  bufferLeft, false, null);
			queue.WriteToBuffer(right, bufferRight, false, null);

			// 引数を設定
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			addEachVector.SetMemoryArgument(0, bufferAnswer);
			addEachVector.SetMemoryArgument(1, bufferLeft);
			addEachVector.SetMemoryArgument(2, bufferRight);

			// 計算を実行
			queue.Execute(addEachVector, null, new long[] { COUNT }, null, null);

			// 結果を読み込み
			queue.ReadFromBuffer(bufferAnswer, ref answer, false, null);

			// 終了まで待機
			queue.Finish();
		}
	}
}