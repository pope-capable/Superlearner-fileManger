import { errorMessage } from 'iyasunday';
import * as service from './service';
const user = {
  id: 1,
  name: 'Balogun Tolu',
};

export async function create(req, res) {
  try {
    req.user = user;
    res.status(200).json(await service.create(req.params, req.body, req.file));
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}

export async function update(req, res) {
  try {
    req.user = user;
    res.status(200).json(await service.update(req.params, req.body, req.file));
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}

export async function view(req, res) {
  try {
    res.status(200).json(await service.view(req.params));
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}

export async function remove(req, res) {
  try {
    res.status(200).json(await service.remove(req.params));
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}

export async function list(req, res) {
  try {
    req.user = user;
    res.status(200).json(await service.list(req.params, req.query));
  } catch (err) {
    res.status(err.httpStatusCode || 500).json(errorMessage(err));
  }
}
