using System.Collections.Generic;
using UnityEngine;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(float[] input)
    {
        //TODO: implement feedworward.
        List<float[,]> thetas= mlpParameters.GetCoeff();//thetas
        List<float[]> sesgos= mlpParameters.GetInter();//sesgos
        //la primera a es la entrada a la que se le añade un 1 delante
        float[] aX = hStack(input);
        //cuantas thetas hay
        int nT = thetas.Count;

        for(int i = 0; i < nT-2;i++)
        {
            //sacamos la z al hacer dot product con la theta y sesgo correspondiente
            float[] zX = dot(aX, thetas[i], sesgos[i]);

            float[] auxA = new float[zX.Length];
            //pasamos la z por la funcion de activacion (sigmodial)
            for (int j =0; j < auxA.Length; j++)
            {
                auxA[j] = sigmoid(zX[j]);
            }
            //se le hace  añade un 1 delante a la nueva a
            auxA = hStack(auxA);

            aX = auxA;
        }
        //sacamos la ultima z al hacer dot product con la theta y sesgo correspondiente
        float[] zL = dot(aX, thetas[nT-1], sesgos[nT - 1]);

        float[] aL = new float[zL.Length];
        //pasamos la ultima z por la funcion de activacion (sigmodial)
        for (int i = 0; i < aL.Length; i++)
        {
            aL[i] = sigmoid(zL[i]);
        }

        //the size of the output layer depends on what actions you have performed in the game.
        //By default it is 7 (number of possible actions) but some actions may not have been performed and therefore the model has assumed that they do not exist.
        //devuelve la ultima a
        return aL;
    }

    private float[] hStack(float[] input) {
        float[] a = new float[input.Length + 1];
        //copia el array que le pasas con un 1 delante
        a[0] = 1;
        for (int i = 0; i < input.Length; i++)
        {
            a[i + 1] = input[i];
        }
        return a;
    }

    private float[] dot(float[] a, float[,] theta, float[] sesgos)
    {
        
        float[] resultado = new float[theta.GetLength(1)];
        //recorre las columnas de la theta
        for (int i = 0; i < theta.GetLength(1); i++)
        {//al primer valor de la a lo multiplicamos por el sesgo correspondiente
            float suma = a[0]*sesgos[i];
            for(int j = 0; j < theta.GetLength(0); j++)
            {//recorre las filas de theta y se multiplica por a y se va sumando
                float mul = a[j+1] * theta[j,i];
                suma += mul;
            }
            resultado[i]=suma;
        }

        return resultado;
    }
    /// <summary>
    /// Calculo de la sigmoidal
    /// </summary>
    /// <param name="z"></param>
    /// <returns></returns>
    private float sigmoid(float z)
    {
        //TODO implementar
        float el = Mathf.Exp(-z);
        float dev = (1 / (1 + el));
        return dev;
    }


    /// <summary>
    /// CAlculo de la soft max, se le pasa el vector de la ulrima capa oculta y devuelve el mismo vector, pero procesado
    /// aplicando softmax a cada uno de los elementos
    /// </summary>
    /// <param name="zArr"></param>
    /// <returns></returns>
    public float[] SoftMax(float[] zArr)
    {
        //TODO implementar
        return zArr;
    }

    /// <summary>
    /// Elige el output de mayor nivel
    /// </summary>
    /// <param name="output"></param>
    /// <returns></returns>
    public int Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        return index;
    }

    /// <summary>
    /// Obtiene el índice de mayor valor.
    /// </summary>
    /// <param name="output"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        for(int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max= output[i];
                index= i;
            }
        }
        //TODO impleemntar.
        return index;
    }
}
