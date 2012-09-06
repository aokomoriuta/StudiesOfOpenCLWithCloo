using System;
using Cloo;

// 倍精度か単精度かどっちか選択
//using Real = System.Single;
using Real = System.Double;

namespace LWisteria.StudiesOfOpenTKWithCloo.Matrix_x_Vector
{
	/// <summary>
	/// メインクラス
	/// </summary>
	static class Matrix_x_VectorMain
	{
		/// <summary>
		/// 行数
		/// </summary>
		const int ROW_COUNT = 1024 * 1024;

		/// <summary>
		/// 1行に含まれる非ゼロ要素の最大列数
		/// </summary>
		const int MAX_NONZERO_COUNT = 30;

		/// <summary>
		/// 全要素数
		/// </summary>
		const int COUNT = ROW_COUNT * MAX_NONZERO_COUNT;

		/// <summary>
		/// 検算値
		/// </summary>
		static Real[] answer;

		/// <summary>
		/// コマンドキュー群
		/// </summary>
		static ComputeCommandQueue[] queues;

		/// <summary>
		/// 各デバイスで計算する要素数
		/// </summary>
		static int[] countPerDevice;

		/// <summary>
		/// デバイスが計算する要素の開始地点
		/// </summary>
		static int[] offset;

		#region バッファー
		/// <summary>
		/// 結果のバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferResult;

		/// <summary>
		/// 行列のバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferMatrix;

		/// <summary>
		/// ベクトルのバッファー
		/// </summary>
		static ComputeBuffer<Real> bufferVector;

		/// <summary>
		/// 列番号のバッファー
		/// </summary>
		static ComputeBuffer<int> bufferColumnIndeces;

		/// <summary>
		/// 非ゼロ要素数のバッファー
		/// </summary>
		static ComputeBuffer<int> bufferNonzeroCount;


		/// <summary>
		/// 行列のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersMatrix;

		/// <summary>
		/// 行列のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersVector;

		/// <summary>
		/// 結果のバッファー群
		/// </summary>
		static ComputeBuffer<Real>[] buffersResult;

		/// <summary>
		/// 列番号のバッファー群
		/// </summary>
		static ComputeBuffer<int>[] buffersColumnIndeces;

		/// <summary>
		/// 非ゼロ要素数のバッファー群
		/// </summary>
		static ComputeBuffer<int>[] buffersNonzeroCount;
		#endregion

		/// <summary>
		/// 行列とベクトルの積を計算するカーネル
		/// </summary>
		static ComputeKernel[] matrix_x_Vector;


		/// <summary>
		/// エントリーポイント
		/// </summary>
		/// <returns>終了コード</returns>
		static int Main()
		{
			Console.WriteLine("= 行列とベクトル積の試験 =");
			Console.WriteLine();

			// 行列の要素、その要素の列番号および非ゼロ要素数を作成
			var matrix = new Real[COUNT];
			var columnIndeces = new int[COUNT];
			var nonzeroCount = new int[ROW_COUNT];

			// ベクトルを作成
			var vector = new Real[ROW_COUNT];

			// 検算値を初期化
			answer = new Real[ROW_COUNT];

			// 全行について
			for(int i = 0; i < ROW_COUNT; i++)
			{
				// 対角成分を初期化
				matrix[i * MAX_NONZERO_COUNT] = 0;

				// 非ゼロ要素数を初期化
				nonzeroCount[i] = 1;

				// 対角成分の列番号を設定
				columnIndeces[i * MAX_NONZERO_COUNT] = i;


				// 各列で
				for(int j = Math.Max(0, i - MAX_NONZERO_COUNT / 2 + 1); j < Math.Min(ROW_COUNT, i + MAX_NONZERO_COUNT / 2); j++)
				{
					if(i != j)
					{
						// 行列の要素を計算
						var a_ij = (Real)(i + j) / 10;//(Real)Math.Abs(Math.Sin(i + j));

						// 要素を設定
						matrix[i * MAX_NONZERO_COUNT + nonzeroCount[i]] = a_ij;

						// 対角成分に追加
						matrix[i * MAX_NONZERO_COUNT] += a_ij;


						// 列番号を設定
						columnIndeces[i * MAX_NONZERO_COUNT + nonzeroCount[i]] = j;

						// 非ゼロ要素数を増やす
						nonzeroCount[i]++;
					}
				}

				// ベクトルの要素を計算して設定
				vector[i] = (Real)i / 10;//(Real)Math.Cos(i) * 10;
			}

			// 行列とベクトルの積を計算して答えに格納
			for(int i = 0; i < ROW_COUNT; i++)
			{
				answer[i] = 0;

				for(int j = 0; j < nonzeroCount[i]; j++)
				{
					answer[i] += matrix[i * MAX_NONZERO_COUNT + j] * vector[columnIndeces[i * MAX_NONZERO_COUNT + j]];
				}
			}

			// CPUの
			foreach(var processor in new System.Management.ManagementClass("Win32_Processor").GetInstances())
			{
				// データを表示
				Console.WriteLine("CPU：{0} ({1}コア)", processor["Name"], processor["NumberOfCores"]);
			}

			// OpenCLの使用準備
			InitializeOpenCL(matrix, vector, nonzeroCount, columnIndeces);

			// 実行開始
			Console.WriteLine();
			Console.WriteLine("== 実行速度計測開始 ==");
			Console.WriteLine("行列サイズ：{0}x{1} ({2}[MB])", ROW_COUNT, MAX_NONZERO_COUNT, COUNT * sizeof(Real) / 1024 / 1024);
			Action<string, bool, Action<Real[]>> showResult = (name, useGpu, action) =>
				Console.WriteLine("{0}: {1,12}", name, ProcessingTime(action, useGpu));

			// 各方法で実行して結果を表示
			showResult("単一CPU", false, (result) => Matrix_x_VectorSingleCpu(result, matrix, vector, nonzeroCount, columnIndeces));
			showResult("複数CPU", false, (result) => Matrix_x_VectorParallelCpu(result, matrix, vector, nonzeroCount, columnIndeces));
			showResult("単一GPU", true, (result) => Matrix_x_VectorSingleGpu(result));
			showResult("複数GPU", true, (result) => { WriteBuffers(matrix, vector, nonzeroCount, columnIndeces); Matrix_x_VectorParallelGpu(result); });

			// もう1回
			Console.WriteLine("---");
			showResult("単一CPU", false, (result) => Matrix_x_VectorSingleCpu(result, matrix, vector, nonzeroCount, columnIndeces));
			showResult("複数CPU", false, (result) => Matrix_x_VectorParallelCpu(result, matrix, vector, nonzeroCount, columnIndeces));
			showResult("単一GPU", true, (result) => Matrix_x_VectorSingleGpu(result));
			showResult("複数GPU", true, (result) => Matrix_x_VectorParallelGpu(result));

			// 成功で終了
			return System.Environment.ExitCode;
		}

		/// <summary>
		/// OpenCL関係の準備をする
		/// </summary>
		static void InitializeOpenCL(Real[] matrix, Real[] vector, int[] nonzeroCount, int[] columnIndeces)
		{
			// プラットフォームを取得
			var platform = ComputePlatform.Platforms[0];
			Console.WriteLine("プラットフォーム：{0} ({1})", platform.Name, platform.Version);

			// コンテキストを作成
			var context = new ComputeContext(Cloo.ComputeDeviceTypes.Gpu, new ComputeContextPropertyList(platform), null, IntPtr.Zero);

			// 利用可能なデバイス群を取得
			var devices = context.Devices;
			Console.WriteLine("デバイス数：{0}", devices.Count);

			// 各デバイスで計算する要素数を初期化
			countPerDevice = new int[devices.Count];

			// 1デバイスが計算する最大要素数を計算
			int maxCountPerDevice = (int)Math.Ceiling((double)ROW_COUNT / devices.Count);

			// デバイスの計算開始番号を作成
			offset = new int[devices.Count];

			// 全デバイスの
			for(int i = 0; i < devices.Count; i++)
			{
				// 計算する要素数を計算
				countPerDevice[i] = maxCountPerDevice - ((i < maxCountPerDevice * devices.Count - ROW_COUNT) ? 1 : 0);

				// 計算開始番号を設定
				offset[i] = (i == 0) ? 0 : (offset[i - 1] + countPerDevice[i - 1]);
			}

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
			var program = new ComputeProgram(context, Properties.Resources.Matrix_x_Vector);

			// ビルドしてみて
			try
			{
				string realString = ((typeof(Real) == typeof(Double)) ? "double" : "float");

				program.Build(devices,
					string.Format(" -D REAL={0} -D MAX_NONZERO_COUNT={1} -Werror", realString, MAX_NONZERO_COUNT),
					null, IntPtr.Zero);
			}
			// 失敗したら
			catch(BuildProgramFailureComputeException ex)
			{
				// ログを表示して例外を投げる
				throw new ApplicationException(string.Format("{0}\n{1}", ex.Message, program.GetBuildLog(devices[0])), ex);
			}

			// カーネルを作成
			matrix_x_Vector = new ComputeKernel[devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				matrix_x_Vector[i] = program.CreateKernel("Matrix_x_Vector");
			}

			// 単一GPU用バッファーを作成
			bufferResult = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadWrite, vector.Length);
			bufferMatrix = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, matrix);
			bufferVector = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, vector);
			bufferColumnIndeces = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, columnIndeces);
			bufferNonzeroCount = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, nonzeroCount);

			// 複数GPU用バッファーを作成
			buffersResult = new ComputeBuffer<Real>[devices.Count];
			buffersMatrix = new ComputeBuffer<Real>[devices.Count];
			buffersVector = new ComputeBuffer<Real>[devices.Count];
			buffersColumnIndeces = new ComputeBuffer<int>[devices.Count];
			buffersNonzeroCount = new ComputeBuffer<int>[devices.Count];
			for(int i = 0; i < devices.Count; i++)
			{
				buffersResult[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.WriteOnly, countPerDevice[i]);
				buffersMatrix[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i] * MAX_NONZERO_COUNT);
				buffersVector[i] = new ComputeBuffer<Real>(context, ComputeMemoryFlags.ReadOnly, ROW_COUNT);
				buffersColumnIndeces[i] = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i] * MAX_NONZERO_COUNT);
				buffersNonzeroCount[i] = new ComputeBuffer<int>(context, ComputeMemoryFlags.ReadOnly, countPerDevice[i]);
			}
		}

		/// <summary>
		/// 操作の実行時間を計測する
		/// </summary>
		/// <param name="action">計測対象の操作</param>
		/// <returns>実行時間（ミリ秒）</returns>
		static long ProcessingTime(Action<Real[]> action, bool useGpu)
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

			// 結果を初期化
			var result = new Real[ROW_COUNT];

			// 計測開始
			stopwatch.Restart();

			// 操作を実行
			action(result);

			// 計測終了
			stopwatch.Stop();

			for(int i = 0; i < result.Length; i++)
			{
				// 精度以上の誤差があれば
				if(result[i] - answer[i] > Math.Pow(10, (Math.Log10(answer[i]) - 8)))
				{
					// 誤差を通知出力
					Console.WriteLine("{3,8}: result={0:e} vs answer={1:e} ({2:e})", result[i], answer[i], result[i] - answer[i], i);
				}
			}

			// 実行時間を返す
			return stopwatch.ElapsedTicks;
		}


		/// <summary>
		/// CPUで並列させずに内積を計算する
		/// </summary>
		/// <param name="result">結果を格納するベクトル</param>
		/// <param name="matrix">行列</param>
		/// <param name="vector">ベクトル</param>
		/// <param name="nonzeroCount">非ゼロ要素数</param>
		/// <param name="columnIndeces">列番号</param>
		static void Matrix_x_VectorSingleCpu(Real[] result, Real[] matrix, Real[] vector, int[] nonzeroCount, int[] columnIndeces)
		{
			// 全行について
			for(int i = 0; i < ROW_COUNT; i++)
			{
				// 結果を初期化
				result[i] = 0;

				// この行の全ての非ゼロ要素数について
				for(int j = 0; j < nonzeroCount[i]; j++)
				{
					// 成分と列番号を取得
					var a_ij = matrix[i * MAX_NONZERO_COUNT + j];
					var columnIndex = columnIndeces[i * MAX_NONZERO_COUNT + j];

					// 結果に格納
					result[i] += a_ij * vector[columnIndex];
				}
			}
		}

		/// <summary>
		/// CPUで並列させてて行列とベクトルの積を計算する
		/// </summary>
		/// <param name="result">結果を格納するベクトル</param>
		/// <param name="matrix">行列</param>
		/// <param name="vector">ベクトル</param>
		/// <param name="nonzeroCount">非ゼロ要素数</param>
		/// <param name="columnIndeces">列番号</param>
		static void Matrix_x_VectorParallelCpu(Real[] result, Real[] matrix, Real[] vector, int[] nonzeroCount, int[] columnIndeces)
		{
			// 全行について並列に
			System.Threading.Tasks.Parallel.For(0, ROW_COUNT, (i) =>
			{
				// 結果を初期化
				result[i] = 0;

				// この行の全ての非ゼロ要素数をについて
				for(int j = 0; j < nonzeroCount[i]; j++)
				{
					// 成分と列番号を取得
					var a_ij = matrix[i * MAX_NONZERO_COUNT + j];
					var columnIndex = columnIndeces[i * MAX_NONZERO_COUNT + j];

					// 結果に格納
					result[i] += a_ij * vector[columnIndex];
				}
			});
		}

		/// <summary>
		///	GPUを1つ使ってて行列とベクトルの積を計算する
		/// </summary>
		/// <param name="result">結果を格納するベクトル</param>
		static void Matrix_x_VectorSingleGpu(Real[] result)
		{
			// 各要素同士の積を計算
			//  # 結果を格納するベクトル
			//  # 計算対象のベクトル1
			//  # 計算対象のベクトル2
			matrix_x_Vector[0].SetMemoryArgument(0, bufferResult);
			matrix_x_Vector[0].SetMemoryArgument(1, bufferMatrix);
			matrix_x_Vector[0].SetMemoryArgument(2, bufferVector);
			matrix_x_Vector[0].SetMemoryArgument(3, bufferColumnIndeces);
			matrix_x_Vector[0].SetMemoryArgument(4, bufferNonzeroCount);
			queues[0].Execute(matrix_x_Vector[0], null, new long[] { ROW_COUNT }, null, null);

			// 結果を読み込み
			queues[0].ReadFromBuffer(bufferResult, ref result, false, null);

			// 終了まで待機
			queues[0].Finish();
		}

		/// <summary>
		///	GPUを複数使ってて行列とベクトルの積を計算する
		/// </summary>
		/// <returns>各要素同士の積の総和</returns>
		static void Matrix_x_VectorParallelGpu(Real[] result)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 各要素同士の積を計算
				//  # 結果を格納するベクトル
				//  # 計算対象のベクトル1
				//  # 計算対象のベクトル2
				matrix_x_Vector[i].SetMemoryArgument(0, buffersResult[i]);
				matrix_x_Vector[i].SetMemoryArgument(1, buffersMatrix[i]);
				matrix_x_Vector[i].SetMemoryArgument(2, buffersVector[i]);
				matrix_x_Vector[i].SetMemoryArgument(3, buffersColumnIndeces[i]);
				matrix_x_Vector[i].SetMemoryArgument(4, buffersNonzeroCount[i]);
				queues[i].Execute(matrix_x_Vector[i], null, new long[] { countPerDevice[i] }, null, null);

				// 結果を読み込み
				queues[i].ReadFromBuffer(buffersResult[i], ref result, false, 0, offset[i], countPerDevice[i], null);

				// 終了まで待機
				queues[i].Finish();
			});
		}

		/// <summary>
		/// 複数GPUに各データを転送する
		/// </summary>
		/// <param name="left">計算対象1</param>
		/// <param name="right">計算対象2</param>
		static void WriteBuffers(Real[] matrix, Real[] vector, int[] nonzeroCount, int[] columnIndeces)
		{
			// 全キューについて
			System.Threading.Tasks.Parallel.For(0, queues.Length, (i) =>
			{
				// 使用するキューを設定
				var queue = queues[i];

				// データを転送
				queue.WriteToBuffer(matrix, buffersMatrix[i], false, offset[i] * MAX_NONZERO_COUNT, 0, countPerDevice[i] * MAX_NONZERO_COUNT, null);
				queue.WriteToBuffer(vector, buffersVector[i], false, 0, 0, ROW_COUNT, null);
				queue.WriteToBuffer(columnIndeces, buffersColumnIndeces[i], false, offset[i] * MAX_NONZERO_COUNT, 0, countPerDevice[i] * MAX_NONZERO_COUNT, null);
				queue.WriteToBuffer(nonzeroCount, buffersNonzeroCount[i], false, offset[i], 0, countPerDevice[i], null);
			});
		}
	}
}