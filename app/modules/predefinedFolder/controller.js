import * as service from './service';
import { errorMessage } from 'iyasunday';

export async function seed(req, res) {
  try {
    res.status(200).json(await service.seed());
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}

export async function list(req, res) {
  try {
    res.status(200).json(await service.list());
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}
